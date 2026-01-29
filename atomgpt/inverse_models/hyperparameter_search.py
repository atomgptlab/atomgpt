#!/usr/bin/env python
# hyperparameter_search.py  (subprocess-per-trial, with preflight guards)

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import optuna
from datasets import load_dataset
from optuna.samplers import TPESampler
from optuna.trial import Trial
from pydantic_settings import BaseSettings
from transformers import TrainerCallback
from peft import PeftModel

# Child-only heavy imports (torch / unsloth / trl) happen inside _run_child_trial()
# so the parent process stays “CUDA-pristine”.

from atomgpt.inverse_models.inverse_models import (
    formatting_prompts_func,
    load_model,
    make_alpaca_json,
    TrainingPropConfig,
)
from atomgpt.inverse_models.loader import FastLanguageModel
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import dumpjson


# ───────────────────────────── Logging ──────────────────────────────
_DEBUG = os.getenv("ATOMGPT_DEBUG", "").lower() in {"1", "true", "yes", "y"}
logging.basicConfig(
    level=logging.DEBUG if _DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hp_search")


# ───────────────────────────── Config ───────────────────────────────
class OptunaSearchConfig(BaseSettings):
    parameters: Dict[str, Dict]
    n_trials: int = 30
    objective_metric: str | List[str] | None = None
    study_direction: str | List[str] | None = None
    time_repeats: int = 1


# ───────────────────────── Metric helpers ───────────────────────────
def last_value(xs: List[float]) -> float:
    return float("inf") if not xs else xs[-1]


def area_under_curve(xs: List[float]) -> float:
    # numpy trapz is deprecated; keep it for now to avoid extra deps
    return float("inf") if not xs else float(np.trapz(xs))


def trend_slope(xs: List[float]) -> float:
    return float("inf") if len(xs) < 2 else abs(np.polyfit(range(len(xs)), xs, 1)[0])


METRIC_EVALUATORS: Dict[str, Callable[[Dict[str, float]], float]] = {
    "training_time": lambda m: m["training_time"],
    "final_train_loss": lambda m: m["final_train_loss"],
    "final_eval_loss": lambda m: m["final_eval_loss"],
    "auc_train_loss": lambda m: m["auc_train_loss"],
    "auc_eval_loss": lambda m: m["auc_eval_loss"],
    "slope_train_loss": lambda m: m["slope_train_loss"],
    "slope_eval_loss": lambda m: m["slope_eval_loss"],
}


def _auto_direction(metric: str) -> str:
    return "maximize" if metric.lower() in {"accuracy", "f1"} else "minimize"


# ───────────────────── Parent-safe seed setting ──────────────────────
def _set_seeds_cpu(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Intentionally NO torch here (parent process must not touch CUDA).


# ───────────────────────── Trial crash type ──────────────────────────
class TrialCrashed(RuntimeError):
    """Raised when a per-trial subprocess exits abnormally (segfault, killed, etc.)."""


# ───────────────────────── Search-space sampler ──────────────────────
class SearchSpaceSampler:
    _SUGGEST = {
        "float": lambda t, k, s: t.suggest_float(k, s["low"], s["high"], log=s.get("log", False)),
        "int": lambda t, k, s: t.suggest_int(k, s["low"], s["high"]),
        "categorical": lambda t, k, s: t.suggest_categorical(k, s["choices"]),
    }

    def __init__(self, space: Dict[str, Dict]):
        self.space = space

    def sample(self, trial: Trial) -> Dict[str, Any]:
        sampled = {}
        for k, spec in self.space.items():
            if not spec.get("include", True) or "condition" in spec:
                continue
            sampled[k] = self._SUGGEST[spec["type"]](trial, k, spec)

        for k, spec in self.space.items():
            cond = spec.get("condition")
            if cond and sampled.get(cond["param"]) == cond["value"]:
                sampled[k] = self._SUGGEST[spec["type"]](trial, k, spec)

        if _DEBUG:
            log.debug("Trial %d — sampled params: %s", trial.number, sampled)
        return sampled


# ───────────────────────── Split helpers ─────────────────────────────
def train_val_test_split_ids(data: List[dict], id_tag: str, seed: int, val_ratio: float, test_ratio: float):
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")
    ids = [r[id_tag] for r in data]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split sizes invalid – make dataset larger or ratios smaller.")

    val_ids = ids[:n_val]
    test_ids = ids[n_val : n_val + n_test]
    train_ids = ids[n_val + n_test :]
    return train_ids, val_ids, test_ids


# ───────────────────────── id_prop.csv loader ────────────────────────
def _load_id_prop_data(id_prop_csv: str, cfg: TrainingPropConfig) -> List[dict]:
    base = Path(id_prop_csv).parent
    with open(id_prop_csv) as fh:
        rows = list(csv.reader(fh))

    records: list[dict] = []
    for row in rows:
        rid, *vals = row
        prop_val = (
            cfg.separator.join(map(str, map(float, vals)))
            if len(vals) > 1
            else str(float(vals[0]))
        )

        fpath = base / rid
        if cfg.file_format == "poscar":
            atoms = Atoms.from_poscar(fpath)
        elif cfg.file_format == "xyz":
            atoms = Atoms.from_xyz(fpath)
        elif cfg.file_format == "pdb":
            atoms = Atoms.from_pdb(fpath)
        else:
            raise ValueError(f"Unsupported file_format '{cfg.file_format}'")

        records.append({cfg.id_tag: rid, cfg.prop: prop_val, "atoms": atoms.to_dict()})
    return records


# ───────────────────────── Guardrails & penalties ────────────────────
def _penalty_for(cfg: TrainingPropConfig, objective_metrics: List[str], base: float = 1e9) -> float | tuple[float, ...]:
    eff = int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps)
    p = base + 1e6 * eff
    return p if len(objective_metrics) == 1 else tuple([p] * len(objective_metrics))


def _apply_overrides(cfg: TrainingPropConfig, overrides: Dict[str, Any]) -> None:
    for k, v in overrides.items():
        setattr(cfg, k, v)


def _model_copy(cfg: TrainingPropConfig) -> TrainingPropConfig:
    # pydantic v2 prefers model_copy; keep backward compatibility
    if hasattr(cfg, "model_copy"):
        return cfg.model_copy(deep=True)
    return cfg.copy(deep=True)


# ───────────────────────── Child trial runner ────────────────────────
def _is_oom_msg(msg: str) -> bool:
    m = msg.lower()
    return ("cuda out of memory" in m) or ("failed to allocate" in m) or ("outofmemoryerror" in m)


def _is_cuda_fatal_msg(msg: str) -> bool:
    m = msg.lower()
    return any(
        s in m
        for s in (
            "illegal memory access",
            "device-side assert",
            "unspecified launch failure",
            "misaligned address",
            "an illegal instruction was encountered",
        )
    )


def _cleanup_cuda_child(torch_mod) -> None:
    gc.collect()
    if getattr(torch_mod, "cuda", None) is None:
        return
    if not torch_mod.cuda.is_available():
        return
    # After some CUDA faults, these may themselves throw; swallow.
    for fn in (torch_mod.cuda.empty_cache, torch_mod.cuda.ipc_collect):
        try:
            fn()
        except Exception:
            pass


def _set_seeds_child(seed: int, torch_mod) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def _train_once_child(cfg: TrainingPropConfig, train_json: Path, val_json: Path) -> Dict[str, float]:
    # Child-only imports
    import torch
    from trl import SFTTrainer, SFTConfig

    model, tok, _ = load_model(path=cfg.model_name, config=cfg)
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
        )

    train_ds = load_dataset("json", data_files=str(train_json), split="train")
    val_ds = load_dataset("json", data_files=str(val_json), split="train")

    fmt = lambda e: formatting_prompts_func(e, cfg.alpaca_prompt)
    train_ds = train_ds.map(fmt, batched=True)
    val_ds = val_ds.map(fmt, batched=True)

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=cfg.seed_val,
        dataset_text_field="text",
        dataset_num_proc=cfg.dataset_num_proc,
        max_seq_length=cfg.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    start = time.perf_counter()
    trainer.train()
    trainer.save_model(cfg.model_save_path)

    runtime = trainer.state.log_history[-1].get("train_runtime", time.perf_counter() - start)

    tl = [e["loss"] for e in trainer.state.log_history if "loss" in e and e.get("step") is None]
    el = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]

    metrics = {
        "training_time": float(runtime),
        "final_train_loss": float(last_value(tl)),
        "final_eval_loss": float(last_value(el)),
        "auc_train_loss": float(area_under_curve(tl)),
        "auc_eval_loss": float(area_under_curve(el)),
        "slope_train_loss": float(trend_slope(tl)),
        "slope_eval_loss": float(trend_slope(el)),
    }

    # cleanup
    del model, tok, trainer
    _cleanup_cuda_child(torch)
    return metrics


def _run_child_trial(payload_path: Path, result_path: Path) -> int:
    """
    Runs exactly one trial. Writes result JSON, then exits.
    Return code:
      0 = handled (ok OR handled failure like OOM/cuda_fatal)
      2 = unhandled python exception (parent will mark trial failed)
    If the process *hard-crashes* (SIGSEGV), parent will see nonzero returncode and/or no result file.
    """
    payload = json.load(open(payload_path))
    cfg_dict = payload["train_cfg"]
    overrides = payload["overrides"]
    train_json = Path(payload["train_json"])
    val_json = Path(payload["val_json"])
    seed = int(payload["seed"])

    # child-only import torch
    import torch

    cfg = TrainingPropConfig(**cfg_dict)
    _apply_overrides(cfg, overrides)

    # isolate outputs into the provided work_dir
    work_dir = Path(payload["work_dir"])
    cfg.output_dir = str(work_dir / "out")
    cfg.model_save_path = str(work_dir / "model")
    cfg.csv_out = str(work_dir / "eval.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        _set_seeds_child(seed, torch)
        metrics = _train_once_child(cfg, train_json, val_json)

        result = {
            "status": "ok",
            "metrics": metrics,
            "effective_batch": int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps),
        }
        result_path.write_text(json.dumps(result))
        return 0

    except Exception as e:
        msg = str(e)
        eff = int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps)
        status = "exception"

        if _is_oom_msg(msg):
            status = "oom"
        elif _is_cuda_fatal_msg(msg):
            status = "cuda_fatal"

        # If it’s OOM/cuda_fatal, we *handle* it and exit(0) so parent can return a penalty.
        # If it’s some other exception, we still write a result file but exit(2) so Optuna can mark FAIL.
        result = {
            "status": status,
            "error": msg,
            "traceback": traceback.format_exc(limit=200),
            "effective_batch": eff,
        }
        try:
            result_path.write_text(json.dumps(result))
        except Exception:
            pass

        _cleanup_cuda_child(torch)
        return 0 if status in {"oom", "cuda_fatal"} else 2


# ───────────────────────── Parent objective ──────────────────────────
def objective(
    trial: Trial,
    train_cfg: TrainingPropConfig,
    hp_cfg: OptunaSearchConfig,
    sampler: SearchSpaceSampler,
    train_json: Path,
    val_json: Path,
    objective_metrics: List[str],
    *,
    max_micro_bs: int = 256,
    max_eff_bs: int = 4096,
    trial_timeout_s: int | None = None,
) -> float | tuple[float, ...]:

    # Keep the parent CUDA-sterile: CPU seeds only
    _set_seeds_cpu(int(train_cfg.seed_val) + int(trial.number))

    cfg = _model_copy(train_cfg)
    overrides = sampler.sample(trial)
    _apply_overrides(cfg, overrides)

    eff = int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps)

    # Preflight guardrails: avoid the grotesque regimes that often trigger CUDA corruption
    if int(cfg.per_device_train_batch_size) > max_micro_bs or eff > max_eff_bs:
        trial.set_user_attr("oom_violation", 1.0)
        trial.set_user_attr("effective_batch", eff)
        trial.set_user_attr("failure", "preflight_cap")
        return _penalty_for(cfg, objective_metrics)

    # Per-trial subprocess sandbox
    work = Path(tempfile.mkdtemp(prefix="optuna_trial_"))
    payload_path = work / "payload.json"
    result_path = work / "result.json"

    payload = {
        "train_cfg": cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict(),
        "overrides": overrides,
        "train_json": str(train_json),
        "val_json": str(val_json),
        "seed": int(train_cfg.seed_val) + int(trial.number),
        "work_dir": str(work),
    }
    payload_path.write_text(json.dumps(payload))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_run_single_trial",
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=None,   # inherit
            stderr=None,   # inherit
            check=False,
            timeout=trial_timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        trial.set_user_attr("failure", "timeout")
        trial.set_user_attr("effective_batch", eff)
        shutil.rmtree(work, ignore_errors=True)
        # Mark FAIL but keep optimization going via catch=(TrialCrashed,...)
        raise TrialCrashed(f"Trial {trial.number} timed out after {trial_timeout_s}s")

    # If the subprocess hard-crashed (SIGSEGV / killed), it may not have written result.json
    if proc.returncode != 0 and not result_path.exists():
        trial.set_user_attr("failure", f"subprocess_returncode_{proc.returncode}")
        trial.set_user_attr("effective_batch", eff)
        shutil.rmtree(work, ignore_errors=True)
        raise TrialCrashed(f"Trial {trial.number} subprocess exited with code {proc.returncode}")

    # If result exists, interpret it
    if result_path.exists():
        res = json.loads(result_path.read_text())
    else:
        res = {"status": "exception", "error": "missing result.json", "effective_batch": eff}

    status = res.get("status", "exception")
    eff2 = int(res.get("effective_batch", eff))
    trial.set_user_attr("effective_batch", eff2)

    try:
        if status == "ok":
            metrics = res["metrics"]

            trial.set_user_attr("oom_violation", 0.0)
            for k, v in metrics.items():
                trial.set_user_attr(k, float(v))

            values = [float(METRIC_EVALUATORS[m](metrics)) for m in objective_metrics]
            return values[0] if len(values) == 1 else tuple(values)

        if status in {"oom", "cuda_fatal"}:
            trial.set_user_attr("oom_violation", 1.0)
            trial.set_user_attr("failure", status)
            trial.set_user_attr("error", res.get("error", ""))
            return _penalty_for(cfg, objective_metrics)

        # Any other python exception: make Optuna mark the trial FAIL (visible),
        # but keep the overall optimization alive via study.optimize(catch=(TrialCrashed,))
        trial.set_user_attr("failure", status)
        trial.set_user_attr("error", res.get("error", ""))
        raise TrialCrashed(f"Trial {trial.number} failed in subprocess: {status}: {res.get('error','')}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ───────────────────────── Main orchestration ────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", help="Path to a TrainingPropConfig JSON")
    # Internal: child runner
    p.add_argument("--_run_single_trial", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--result", type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    # Child mode
    if args._run_single_trial:
        if not args.payload or not args.result:
            raise SystemExit("Child mode requires --payload and --result")
        rc = _run_child_trial(Path(args.payload), Path(args.result))
        raise SystemExit(rc)

    # Parent mode
    if not args.config_name:
        raise SystemExit("--config_name is required")

    train_cfg = TrainingPropConfig(**json.load(open(args.config_name)))
    hp_cfg = OptunaSearchConfig(**json.load(open(train_cfg.hp_cfg_path)))

    obj = hp_cfg.objective_metric or "final_eval_loss"
    objective_metrics = [obj] if isinstance(obj, str) else list(obj)

    dirs = hp_cfg.study_direction
    if dirs is None:
        directions = [_auto_direction(k) for k in objective_metrics]
    else:
        directions = [dirs] if isinstance(dirs, str) else list(dirs)

    if len(directions) == 1 and len(objective_metrics) > 1:
        directions = directions * len(objective_metrics)

    if _DEBUG:
        log.debug("Objectives: %s | Directions: %s", objective_metrics, directions)

    # Build dataset JSONs once (shared across trials)
    data = _load_id_prop_data(train_cfg.id_prop_path, train_cfg)

    train_ids, val_ids, test_ids = train_val_test_split_ids(
        data,
        train_cfg.id_tag,
        train_cfg.seed_val,
        train_cfg.val_ratio,
        train_cfg.test_ratio,
    )

    tmp = Path(tempfile.mkdtemp(prefix="optuna_data_"))
    train_j = tmp / "train.json"
    val_j = tmp / "val.json"
    test_j = tmp / "test.json"
    dumpjson(make_alpaca_json(data, train_ids, config=train_cfg), train_j)
    dumpjson(make_alpaca_json(data, val_ids, config=train_cfg), val_j)
    dumpjson(make_alpaca_json(data, test_ids, config=train_cfg), test_j)

    sampler = SearchSpaceSampler(hp_cfg.parameters)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    opt_sampler = TPESampler(
        multivariate=True,
        constraints_func=lambda t: (t.user_attrs.get("oom_violation", 0.0),),
    )
    study = optuna.create_study(directions=directions, pruner=pruner, sampler=opt_sampler)

    wall = time.time()
    try:
        study.optimize(
            partial(
                objective,
                train_cfg=train_cfg,
                hp_cfg=hp_cfg,
                sampler=sampler,
                train_json=train_j,
                val_json=val_j,
                objective_metrics=objective_metrics,
                # GUARDS — adjust to taste
                max_micro_bs=256,
                max_eff_bs=4096,
                trial_timeout_s=None,
            ),
            n_trials=hp_cfg.n_trials,
            # CRITICAL: this keeps the *study* alive while marking the trial FAIL (visible)
            catch=(TrialCrashed,),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    runtime = time.time() - wall
    print(f"\nStudy finished in {runtime:.1f}s")
    if len(objective_metrics) == 1:
        print("Best value :", study.best_value)
        print("Best params:", study.best_params)
    else:
        print("Pareto front (top 5 shown):")
        for i, t in enumerate(study.best_trials[:5]):
            print(f"  Trial {t.number}: values={t.values}, params={t.params}")


if __name__ == "__main__":
    main()
