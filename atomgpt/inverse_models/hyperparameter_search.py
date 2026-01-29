#!/usr/bin/env python
# hyperparameter_search.py

from __future__ import annotations

import argparse, json, os, random, shutil, tempfile, time, csv, logging, gc
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Dict, List, Callable, Any

import numpy as np, optuna, torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datasets import load_dataset
from optuna.trial import Trial
from pydantic_settings import BaseSettings
from transformers import IntervalStrategy, TrainingArguments, TrainerCallback
from peft import PeftModel

from atomgpt.inverse_models.loader import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from atomgpt.inverse_models.inverse_models import (
    evaluate,
    make_alpaca_json,
    formatting_prompts_func,
    load_model,
    TrainingPropConfig
)
from jarvis.db.jsonutils import dumpjson, loadjson
from jarvis.core.atoms import Atoms

# ═════════════════════════════ Logging ══════════════════════════════
"""
export ATOMGPT_DEBUG="true" to see debug lines in the console
"""
_DEBUG = os.getenv("ATOMGPT_DEBUG", "").lower() in {"1", "true", "yes", "y"}
logging.basicConfig(
    level=logging.DEBUG if _DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hp_search")

# ═════════════════════════════ Config ═══════════════════════════════
"""
OptunaSearchConfig is a schema for defining hyperparemeters 
and values specific to a hyperparameter search study conducted 
with this script
"""
class OptunaSearchConfig(BaseSettings):
    parameters: Dict[str, Dict]
    n_trials: int = 30
    objective_metric: str | List[str] | None = None
    study_direction: str | List[str] | None = None
    time_repeats: int = 1


# ═════════════════════════ Metrics helpers ═══════════════════════════
"""
last_value() returns the final value of the optimization parameter.
"""
def last_value(xs: List[float]) -> float:
    return float("inf") if not xs else xs[-1]


"""
area_under_curve() returns the area under the curve of a set of 
optimization parameter values. 
"""
def area_under_curve(xs: List[float]) -> float:
    return float("inf") if not xs else np.trapz(xs)


"""
trend_slope() returns the slope of the line of best fit for a
set of optimization parameter values.
"""
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


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in msg
        or "failed to allocate" in msg
        or ("cublas" in msg and "alloc" in msg)
    )


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ═════════════════════ Search-space sampler ══════════════════════════
class SearchSpaceSampler:
    _SUGGEST = {
        "float": lambda t, k, s: t.suggest_float(
            k, s["low"], s["high"], log=s.get("log", False)
        ),
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


# ═════════════════════ Optuna pruning callback ═══════════════════════
"""
Ends the current trial if deemed unpromising.
"""
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: Trial, key: str):
        self.trial, self.key = trial, key

    def on_evaluate(self, *_, metrics=None, **__):
        if metrics and self.key in metrics:
            step = metrics.get("epoch", 0)
            self.trial.report(metrics[self.key], step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


# ═════════════════════ Split helpers ═════════════════════════════════
def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


"""
Deterministically shuffle IDs and split into train/val/test lists.
"""
def train_val_test_split_ids(
    data: List[dict], id_tag: str, seed: int, val_ratio: float, test_ratio: float
):
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


# ═════════════════════ Single train-pass ═════════════════════════════
def _train_once(
    cfg: TrainingPropConfig,
    train_json: Path,
    val_json: Path,
    prune_cb: TrainerCallback | None,
) -> Dict[str, float]:

    model, tok, _ = load_model(path=cfg.model_name, config=cfg)
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
        )
    # ─── datasets ────────────────────────────────────────────────────
    train_ds = load_dataset("json", data_files=str(train_json), split="train")
    val_ds = load_dataset("json", data_files=str(val_json), split="train")

    fmt = lambda e: formatting_prompts_func(e, cfg.alpaca_prompt)
    train_ds = train_ds.map(fmt, batched=True)
    val_ds = val_ds.map(fmt, batched=True)

    if _DEBUG:
        log.debug(
            "Prepared datasets — train: %d samples, val: %d samples",
            len(train_ds),
            len(val_ds),
        )

    sft_args = SFTConfig(
        # --- training ---
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

        # --- data prep (must live in config for AtomGPT wrapper) ---
        dataset_text_field="text",
        dataset_num_proc=cfg.dataset_num_proc,
        max_seq_length=cfg.max_seq_length,
        packing=False,            # or True if you really want packing; wrapper warns about it
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        processing_class=tok,     # <- replaces tokenizer=tok
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # (optional) compute_metrics=..., callbacks=..., peft_config=..., formatting_func=...
    )
    if prune_cb:
        trainer.add_callback(prune_cb)

    start = time.perf_counter()
    trainer.train()
    trainer.save_model(cfg.model_save_path)

    runtime = trainer.state.log_history[-1].get(
        "train_runtime", time.perf_counter() - start
    )

    tl = [
        e["loss"]
        for e in trainer.state.log_history
        if "loss" in e and e.get("step") is None
    ]
    el = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]

    metrics = {
        "training_time": runtime,
        "final_train_loss": last_value(tl),
        "final_eval_loss": last_value(el),
        "auc_train_loss": area_under_curve(tl),
        "auc_eval_loss": area_under_curve(el),
        "slope_train_loss": trend_slope(tl),
        "slope_eval_loss": trend_slope(el),
    }
    if _DEBUG:
        log.debug("Single pass metrics: %s", metrics)

    del model, tok, trainer
    torch.cuda.empty_cache()
    return metrics


# ═════════════════════ Optuna objective ══════════════════════════════
def objective(
    trial: Trial,
    train_cfg: TrainingPropConfig,
    hp_cfg: OptunaSearchConfig,
    sampler: SearchSpaceSampler,
    train_json: Path,
    val_json: Path,
    test_json: Path,
    objective_metrics: List[str],
) -> List[float] | float:

    _set_seeds(train_cfg.seed_val + trial.number)
    cfg = train_cfg.copy(deep=True)

    for k, v in sampler.sample(trial).items():
        setattr(cfg, k, v)

    work = Path(tempfile.mkdtemp(prefix="optuna_"))
    cfg.output_dir = str(work / "out")
    cfg.model_save_path = str(work / "model")
    cfg.csv_out = str(work / "eval.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        metrics_avgs = []
        for _ in range(hp_cfg.time_repeats):
            prune_cb = OptunaPruningCallback(trial, objective_metrics[0])
            try:
                metrics_avgs.append(_train_once(cfg, train_json, val_json, prune_cb))
            except Exception as e:
                if _is_oom(e):
                    eff = int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps)
                    trial.set_user_attr("oom_violation", 1.0)
                    trial.set_user_attr("effective_batch", eff)
                    _cleanup_cuda()
                    penalty = 1e9 + 1e6 * eff
                    return penalty if len(objective_metrics) == 1 else tuple([penalty] * len(objective_metrics))
                raise

        metrics = {
            k: float(np.mean([d[k] for d in metrics_avgs])) for k in metrics_avgs[0]
        }

        model, tok = FastLanguageModel.from_pretrained(cfg.model_save_path)

        trial.set_user_attr("oom_violation", 0.0)
        trial.set_user_attr("effective_batch", int(cfg.per_device_train_batch_size) * int(cfg.gradient_accumulation_steps))

        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("metrics_vec", [metrics[m] for m in objective_metrics])

        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        with open(log_path / "optuna_trials.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {"number": trial.number, "params": trial.params, "metrics": metrics}
                )
                + "\n"
            )

        if _DEBUG:
            log.debug("Trial %d finished — metrics: %s", trial.number, metrics)

        out = [METRIC_EVALUATORS[m](metrics) for m in objective_metrics]
        return out[0] if len(out) == 1 else tuple(out)

    finally:
        shutil.rmtree(work, ignore_errors=True)
        _cleanup_cuda()


# ═══════════════════ id_prop.csv loader  ════════════════════
def _load_id_prop_data(id_prop_csv: str, cfg: TrainingPropConfig) -> List[dict]:
    """
    Read a standard id_prop.csv file and accompanying structure files,
    returning records compatible with `make_alpaca_json`.
    """
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

        records.append(
            {
                cfg.id_tag: rid,
                cfg.prop: prop_val,
                "atoms": atoms.to_dict(),
            }
        )
    return records


# ═════════════════════ CLI / study orchestration ═════════════════════
def main() -> None:
    """
    Entrypoint: run an Optuna HPO study for AtomGPT fine-tuning.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True, help="Path to a TrainingPropConfig JSON")
    args = p.parse_args()

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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    data = _load_id_prop_data(train_cfg.id_prop_path, train_cfg)

    train_ids, val_ids, test_ids = train_val_test_split_ids(
        data,
        train_cfg.id_tag,
        train_cfg.seed_val,
        train_cfg.val_ratio,
        train_cfg.test_ratio,
    )

    if _DEBUG:
        log.debug(
            "Dataset split sizes — train: %d | val: %d | test: %d",
            len(train_ids),
            len(val_ids),
            len(test_ids),
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
    study.optimize(
        partial(
            objective,
            train_cfg=train_cfg,
            hp_cfg=hp_cfg,
            sampler=sampler,
            train_json=train_j,
            val_json=val_j,
            test_json=test_j,
            objective_metrics=objective_metrics,
        ),
        n_trials=hp_cfg.n_trials,
    )
    runtime = time.time() - wall
    print("\nStudy finished in %.1fs" % runtime)
    if len(objective_metrics) == 1:
        print("Best value :", study.best_value)
        print("Best params:", study.best_params)
    else:
        print("Pareto front (top 5 shown):")
        for i, t in enumerate(study.best_trials[:5]):
            print(f"  Trial {t.number}: values={t.values}, params={t.params}")

    if _DEBUG:
        log.debug("Full study completed in %.1fs", runtime)


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
