# preshard_dataset.py

import argparse
import csv
import json
import os
import time
from typing import Optional, Literal

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoTokenizer

from jarvis.core.atoms import Atoms
from jarvis.core.composition import Composition

from jarvis.db.jsonutils import loadjson
from atomgpt.inverse_models.utils import get_crystal_string_t, get_figlet
from atomgpt.inverse_models.inverse_models import TrainingPropConfig, get_input


parser = argparse.ArgumentParser(description="Pretokenize AtomGPT inverse-model dataset.")
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)


class PretokConfig(BaseModel):
    id_prop_path: str
    tokenizer_class: str
    model_name: str
    max_seq_length: int
    num_train: Optional[int] = None
    num_val: Optional[int] = None
    num_test: Optional[int] = None
    test_ratio: Optional[float] = None
    val_ratio: Optional[float] = None
    separator: str = ","
    prop: str
    id_tag: str = "id"
    file_format: Literal["poscar", "xyz", "cif", "pdb"] = "poscar"
    instruction: str
    alpaca_prompt: str
    output_prompt: str
    chem_info: Literal["none", "formula", "element_list", "element_dict"] = "formula"
    dataset_num_proc: int = 1
    dataloader_num_workers: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    seed_val: int = 3407


def _save_to_disk(ds, path: str, max_shard_size: str = "1GB"):
    os.makedirs(path, exist_ok=True)
    try:
        ds.save_to_disk(path, max_shard_size=max_shard_size)
    except TypeError:
        ds.save_to_disk(path)


def _count_rows(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _compute_split_sizes(cfg: TrainingPropConfig, raw_cfg_dict: dict, n_all: int):
    num_train = cfg.num_train
    num_test = cfg.num_test

    test_ratio = cfg.test_ratio if cfg.test_ratio is not None else 0.0
    val_ratio = raw_cfg_dict.get("val_ratio", getattr(cfg, "val_ratio", 0.1))
    if val_ratio is None:
        val_ratio = 0.1

    if not num_train:
        num_test = int(n_all * test_ratio)
        num_val = int(n_all * val_ratio)
        if num_val <= 0 and val_ratio > 0 and num_test > 0:
            num_val = 1
            num_test = max(0, num_test - 1)
        num_train = n_all - num_test - num_val
    else:
        num_val = int(n_all * val_ratio)
        if num_test is None:
            num_test = int(n_all * test_ratio)
        if num_train + num_val + (num_test or 0) > n_all:
            num_val = max(0, n_all - num_train - (num_test or 0))

    return int(num_train), int(num_val), int(num_test or 0), float(val_ratio), float(test_ratio)


def _atoms_from_file(run_path: str, jid: str, file_format: str) -> Atoms:
    pth = os.path.join(run_path, jid)
    if file_format == "poscar":
        return Atoms.from_poscar(pth)
    if file_format == "xyz":
        return Atoms.from_xyz(pth)
    if file_format == "cif":
        return Atoms.from_cif(pth)
    if file_format == "pdb":
        return Atoms.from_pdb(pth)
    raise ValueError(f"Unsupported file_format={file_format}")


def _chem_string(cfg: TrainingPropConfig, atoms: Atoms) -> str:
    if cfg.chem_info == "none":
        return ""
    if cfg.chem_info == "element_list":
        return atoms.composition.search_string
    if cfg.chem_info == "element_dict":
        comp = Composition.from_string(atoms.composition.reduced_formula)
        return str(dict(sorted(comp.to_dict().items())))
    return atoms.composition.reduced_formula


def _parse_prop(row, separator: str):
    vals = row[1:]
    if not vals:
        return "na"
    if any(str(v).strip().lower() == "na" for v in vals):
        return "na"
    try:
        if len(vals) == 1:
            return str(float(vals[0]))
        return separator.join(map(str, [float(v) for v in vals]))
    except Exception:
        if len(vals) == 1:
            return str(vals[0])
        return separator.join(map(str, vals))


class _ShardWriter:
    def __init__(self, out_dir: str, split: str, records_per_shard: int = 100_000):
        self.out_dir = out_dir
        self.split = split
        self.records_per_shard = records_per_shard
        self.part = 0
        self.n_in_part = 0
        self.files = []
        self._fh = None
        self._open_new()

    def _open_new(self):
        if self._fh:
            self._fh.close()
        fname = os.path.join(self.out_dir, f"{self.split}-{self.part:05d}.jsonl")
        self.files.append(fname)
        self._fh = open(fname, "w")
        self.part += 1
        self.n_in_part = 0

    def write(self, obj: dict):
        if self.n_in_part >= self.records_per_shard:
            self._open_new()
        self._fh.write(json.dumps(obj) + "\n")
        self.n_in_part += 1

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ---- multiprocessing worker plumbing (added) ----
_WORKER_CFG = None
_WORKER_RUN_PATH = None
_WORKER_EOS = None


def _worker_init(raw_cfg: dict, run_path: str, eos_token: str):
    global _WORKER_CFG, _WORKER_RUN_PATH, _WORKER_EOS
    _WORKER_CFG = TrainingPropConfig(**raw_cfg)
    _WORKER_RUN_PATH = run_path
    _WORKER_EOS = eos_token


def _build_record_worker(jid: str, prop_val: str) -> dict:
    cfg = _WORKER_CFG
    atoms = _atoms_from_file(_WORKER_RUN_PATH, jid, cfg.file_format)
    chem = _chem_string(cfg, atoms)
    inp = get_input(config=cfg, chem=chem, val=prop_val)
    out = get_crystal_string_t(atoms)
    text = cfg.alpaca_prompt.format(cfg.instruction, inp, out) + _WORKER_EOS
    return {
        "id": jid,
        "instruction": cfg.instruction,
        "input": inp,
        "output": out,
        "text": text,
    }
# -----------------------------------------------


def main(config_file=None):
    if config_file is None:
        args = parser.parse_args()
        config_file = args.config_name

    print(get_figlet())
    t0 = time.time()

    raw_cfg = loadjson(config_file)
    cfg = TrainingPropConfig(**raw_cfg)

    id_prop_path = cfg.id_prop_path
    run_path = os.path.dirname(id_prop_path)

    base_dir = os.path.dirname(os.path.abspath(id_prop_path))
    pretok_dir = os.path.join(base_dir, str(cfg.tokenizer_class))
    os.makedirs(pretok_dir, exist_ok=True)

    pretok_max_len = int(raw_cfg.get("max_seq_length", cfg.max_seq_length))
    if pretok_max_len <= 0:
        raise ValueError(f"Invalid max_seq_length={pretok_max_len}")

    n_all = _count_rows(id_prop_path)
    num_train, num_val, num_test, val_ratio, test_ratio = _compute_split_sizes(cfg, raw_cfg, n_all)

    print("n_all", n_all)
    print("num_train", num_train)
    print("num_val", num_val)
    print("num_test", num_test)
    print("max_seq_length", pretok_max_len)

    pretok_meta = PretokConfig(
        id_prop_path=id_prop_path,
        tokenizer_class=cfg.tokenizer_class,
        model_name=cfg.model_name,
        max_seq_length=pretok_max_len,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        separator=cfg.separator,
        prop=cfg.prop,
        id_tag=cfg.id_tag,
        file_format=cfg.file_format,
        instruction=cfg.instruction,
        alpaca_prompt=cfg.alpaca_prompt,
        output_prompt=cfg.output_prompt,
        chem_info=cfg.chem_info,
        dataset_num_proc=cfg.dataset_num_proc,
        dataloader_num_workers=getattr(cfg, "dataloader_num_workers", 1),
        per_device_train_batch_size=getattr(cfg, "per_device_train_batch_size", 2),
        per_device_eval_batch_size=getattr(cfg, "per_device_eval_batch_size", 2),
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 1),
        seed_val=getattr(cfg, "seed_val", 3407),
    )
    with open(os.path.join(pretok_dir, "pretok_metadata.json"), "w") as f:
        f.write(json.dumps(pretok_meta.dict(), indent=2))

    records_per_shard = int(os.environ.get("PRETOK_RECORDS_PER_SHARD", "100000"))
    w_train = _ShardWriter(pretok_dir, "train", records_per_shard=records_per_shard)
    w_val = _ShardWriter(pretok_dir, "val", records_per_shard=records_per_shard)
    w_test = _ShardWriter(pretok_dir, "test", records_per_shard=records_per_shard)

    EOS_TOKEN = "</s>"
    n_target = num_train + num_val + num_test

    # choose record-building workers from Slurm/env (added)
    record_num_proc = int(
        os.environ.get(
            "PRETOK_RECORD_NUM_PROC",
            os.environ.get("SLURM_CPUS_PER_TASK", "1"),
        )
    )
    if record_num_proc < 1:
        record_num_proc = 1
    max_in_flight = int(os.environ.get("PRETOK_MAX_INFLIGHT", str(record_num_proc * 4)))

    try:
        with open(id_prop_path, "r") as f:
            reader = csv.reader(f)

            if record_num_proc <= 1:
                # original serial behavior
                for idx, row in enumerate(tqdm(reader, total=n_all)):
                    if idx >= n_target:
                        break
                    if not row:
                        continue
                    jid = row[0]

                    if idx < num_train:
                        writer = w_train
                    elif idx < num_train + num_val:
                        writer = w_val
                    elif idx < num_train + num_val + num_test:
                        writer = w_test
                    else:
                        break

                    prop_val = _parse_prop(row, separator=cfg.separator)
                    if str(prop_val).strip().lower() == "na":
                        continue

                    atoms = _atoms_from_file(run_path, jid, cfg.file_format)
                    chem = _chem_string(cfg, atoms)
                    inp = get_input(config=cfg, chem=chem, val=prop_val)
                    out = get_crystal_string_t(atoms)
                    text = cfg.alpaca_prompt.format(cfg.instruction, inp, out) + EOS_TOKEN

                    writer.write(
                        {
                            "id": jid,
                            "instruction": cfg.instruction,
                            "input": inp,
                            "output": out,
                            "text": text,
                        }
                    )

            else:
                # parallel record construction; deterministic flush in idx order (added)
                buffer = {}
                next_idx = 0
                futures = {}

                def _flush_ready():
                    nonlocal next_idx
                    while next_idx in buffer:
                        rec = buffer.pop(next_idx)
                        if rec is not None:
                            if next_idx < num_train:
                                w = w_train
                            elif next_idx < num_train + num_val:
                                w = w_val
                            else:
                                w = w_test
                            w.write(rec)
                        next_idx += 1

                with ProcessPoolExecutor(
                    max_workers=record_num_proc,
                    initializer=_worker_init,
                    initargs=(raw_cfg, run_path, EOS_TOKEN),
                ) as ex:
                    for idx, row in enumerate(tqdm(reader, total=n_all)):
                        if idx >= n_target:
                            break
                        if not row:
                            buffer[idx] = None
                            _flush_ready()
                            continue

                        jid = row[0]
                        prop_val = _parse_prop(row, separator=cfg.separator)
                        if str(prop_val).strip().lower() == "na":
                            buffer[idx] = None
                            _flush_ready()
                            continue

                        while len(futures) >= max_in_flight:
                            done, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for fut in done:
                                i_done = futures.pop(fut)
                                buffer[i_done] = fut.result()
                            _flush_ready()

                        fut = ex.submit(_build_record_worker, jid, prop_val)
                        futures[fut] = idx

                    for fut in as_completed(list(futures.keys())):
                        i_done = futures[fut]
                        buffer[i_done] = fut.result()
                        _flush_ready()

                _flush_ready()

    finally:
        w_train.close()
        w_val.close()
        w_test.close()

    data_files = {
        "train": w_train.files,
        "validation": w_val.files,
        "test": w_test.files,
    }

    ds = load_dataset("json", data_files=data_files)

    _save_to_disk(ds, os.path.join(pretok_dir, "alpaca"), max_shard_size="1GB")
    _save_to_disk(ds.select_columns(["id", "text"]), os.path.join(pretok_dir, "text"), max_shard_size="1GB")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=pretok_max_len,
            padding=False,
        )

    tok_ds = ds.map(
        tok_fn,
        batched=True,
        num_proc=cfg.dataset_num_proc,
        remove_columns=["instruction", "input", "output", "text"],
    )
    _save_to_disk(tok_ds, os.path.join(pretok_dir, "tokenized"), max_shard_size="1GB")

    print("Done. Seconds:", time.time() - t0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(config_file=args.config_name)
