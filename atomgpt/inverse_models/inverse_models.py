from typing import Optional
from atomgpt.inverse_models.loader import FastLanguageModel

from atomgpt.inverse_models.callbacks import (
    PrintGPUUsageCallback,
    ExampleTrainerCallback,
)
from transformers import (
    TrainingArguments,
)
import torch
from atomgpt.inverse_models.utils import (
    gen_atoms,
    text2atoms,
    get_crystal_string_t,
    get_figlet,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel
from datasets import load_dataset
from functools import partial
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from tqdm import tqdm
import pprint
from jarvis.io.vasp.inputs import Poscar
import csv
import os
import numpy as np
from pydantic_settings import BaseSettings
import sys
import json
import argparse
from typing import Literal
import time
from jarvis.core.composition import Composition

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)


class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    id_prop_path: Optional[str] = "atomgpt/examples/inverse_model/id_prop.csv"
    prefix: str = "atomgpt_run"
    model_name: str = "knc6/atomgpt_mistral_tc_supercon"
    batch_size: int = 2
    num_epochs: int = 2
    logging_steps: int = 1
    dataset_num_proc: int = 2
    seed_val: int = 3407
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train: Optional[int] = None
    num_test: Optional[int] = None
    test_ratio: Optional[float] = 0.1
    val_ratio: Optional[float] = 0.1
    model_save_path: str = "atomgpt_lora_model"
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    loss_type: str = "default"
    optim: str = "adamw_8bit"
    id_tag: str = "id"
    lr_scheduler_type: str = "linear"
    separator: str = ","
    prop: str = "Tc_supercon"
    output_dir: str = "outputs"
    csv_out: str = "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    chem_info: Literal["none", "formula", "element_list", "element_dict"] = (
        "formula"
    )
    file_format: Literal["poscar", "xyz", "pdb"] = "poscar"
    save_strategy: Literal["epoch", "steps", "no"] = "steps"
    save_steps: int = 2
    evaluation_strategy: Literal["epoch", "steps", "no"] = "steps"
    eval_steps: int = 2
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    save_total_limit: Optional[int] = None
    callback_samples: int = 2
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    instruction: str = "Below is a description of a superconductor material."
    alpaca_prompt: str = (
        "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
    )
    output_prompt: str = (
        " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
    )
    hp_cfg_path: Optional[str] = "hp_search_config.json"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 3
    warmup_ratio: float = 0.0
    logging_steps: int = 10


def get_input(config=None, chem="", val=10):
    if config.chem_info == "none":
        prefix = ""
    elif config.chem_info == "element_list":
        prefix = (
            "The chemical elements are "
            + chem
            + " . "
        )
    elif config.chem_info == "element_dict":
        prefix = (
            "The chemical contents are "
            + chem
            + " . "
        )
    elif config.chem_info == "formula":
        prefix = (
            "The chemical formula is "
            + chem
            + " . "
        )

    inp = (
        prefix
        + "The  "
        + config.prop
        + " is "
        + str(val)
        + "."
        + config.output_prompt
    )
    return inp


def make_alpaca_json(
    dataset=[],
    jids=[],
    include_jid=False,
    config=None,
):
    mem = []
    print("config.prop", config.prop)
    for i in dataset:
        if i[config.prop] != "na" and i[config.id_tag] in jids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            if include_jid:
                info["id"] = i[config.id_tag]
            info["instruction"] = config.instruction
            if config.chem_info == "none":
                chem = ""
            elif config.chem_info == "element_list":
                chem = atoms.composition.search_string
            elif config.chem_info == "element_dict":
                comp = Composition.from_string(
                    atoms.composition.reduced_formula
                )
                chem = comp.to_dict()
                chem = str(dict(sorted(chem.items())))
            elif config.chem_info == "formula":
                chem = atoms.composition.reduced_formula

            inp = get_input(config=config, val=i[config.prop], chem=chem)
            info["input"] = inp

            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


def formatting_prompts_func(examples, alpaca_prompt):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = "</s>"
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


def load_model(path="", config=None):
    if config is None:
        config_file = os.path.join(path, "config.json")
        config = loadjson(config_file)
        config = TrainingPropConfig(**config)
        pprint.pprint(config.dict())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, config


def _validate_atoms(atoms):
    if atoms is None:
        return False, "atoms_is_none"
    try:
        lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
        if lat.shape != (3, 3):
            return False, f"bad_lattice_shape:{getattr(atoms,'lattice_mat',None)}"
        if not np.isfinite(lat).all():
            return False, "nonfinite_lattice"
        n = getattr(atoms, "num_atoms", None)
        if n is None or n <= 0:
            return False, f"num_atoms_invalid:{n}"
        _ = Poscar(atoms).to_string()
        return True, ""
    except Exception as e:
        return False, f"poscar_fail:{type(e).__name__}:{e}"


def _poscar_one_line(at):
    return Poscar(at).to_string().replace("\n", "\\n")


def _misses_path(csv_out, config):
    fname = getattr(config, "miss_csv", None)
    if fname is None or not str(fname).strip():
        root, ext = os.path.splitext(csv_out)
        fname = root + ".misses.csv"
    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    return fname


def evaluate(
    test_set=[],
    model="",
    tokenizer="",
    csv_out="out.csv",
    config="",
):
    print("Testing\n", len(test_set))
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
    miss_csv_out = _misses_path(csv_out, config)

    with open(csv_out, "w", newline="") as f_ok, open(miss_csv_out, "w", newline="") as f_miss:
        ok_writer = csv.writer(f_ok)
        miss_writer = csv.writer(f_miss)
        ok_writer.writerow(["id", "target", "prediction"])
        miss_writer.writerow(["id", "stage", "error", "detail", "raw_text_preview"])

        for i in tqdm(test_set, total=len(test_set)):
            sample_id = i.get("id", "")
            target_mat = None
            target_err = None
            try:
                target_mat = text2atoms("\n" + i["output"])
                ok, detail = _validate_atoms(target_mat)
                if not ok:
                    target_err = detail
            except Exception as e:
                target_err = f"text2atoms:{type(e).__name__}:{e}"

            if target_err:
                miss_writer.writerow([sample_id, "target", "invalid_target", target_err, (i.get("output", "")[:240])])
                continue

            gen_mat = None
            gen_err = None
            try:
                gen_mat = gen_atoms(
                    prompt=i["input"],
                    tokenizer=tokenizer,
                    model=model,
                    alpaca_prompt=config.alpaca_prompt,
                    instruction=config.instruction,
                )
                ok, detail = _validate_atoms(gen_mat)
                if not ok:
                    gen_err = detail
            except Exception as e:
                gen_err = f"gen_atoms:{type(e).__name__}:{e}"

            if gen_err:
                miss_writer.writerow([sample_id, "prediction", "invalid_prediction", gen_err, ""])
                continue

            try:
                ok_writer.writerow([
                    sample_id,
                    _poscar_one_line(target_mat),
                    _poscar_one_line(gen_mat),
                ])
            except Exception as e:
                miss_writer.writerow([sample_id, "write", "write_failed", f"{type(e).__name__}:{e}", ""])


def main(config_file=None):
    if config_file is None:
        args = parser.parse_args(sys.argv[1:])
        config_file = args.config_name
    if not torch.cuda.is_available():
        raise ValueError("Currently model training is possible with GPU only.")
    figlet = get_figlet()
    print(figlet)
    t1 = time.time()
    print("config_file", config_file)
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config.dict())
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    f = open(os.path.join(config.model_save_path, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    id_prop_path = config.id_prop_path
    run_path = os.path.dirname(id_prop_path)
    num_train = config.num_train
    num_test = config.num_test
    callback_samples = config.callback_samples
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    dat = []
    ids = []
    for i in tqdm(dt, total=len(dt)):
        info = {}
        info["id"] = i[0]
        ids.append(i[0])
        tmp = [j for j in i[1:]]
        if len(tmp) == 1:
            tmp = str(float(tmp[0]))
        else:
            tmp = config.separator.join(map(str, tmp))

        info[config.prop] = tmp
        pth = os.path.join(run_path, info["id"])
        if config.file_format == "poscar":
            atoms = Atoms.from_poscar(pth)
        elif config.file_format == "xyz":
            atoms = Atoms.from_xyz(pth)
        elif config.file_format == "cif":
            atoms = Atoms.from_cif(pth)
        elif config.file_format == "pdb":
            atoms = Atoms.from_pdb(pth)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    n_total = len(ids)
    if num_train is None and num_test is None:
        num_test = int(n_total * (config.test_ratio or 0.0))
        num_val = int(n_total * (config.val_ratio or 0.0))
        num_train = n_total - num_test - num_val
    else:
        if num_train is None:
            num_train = n_total - (num_test or 0)
        if num_test is None:
            num_test = max(0, n_total - num_train)
        num_val = max(0, n_total - num_train - num_test)

    train_ids = ids[0:num_train]
    val_ids = ids[num_train:num_train + num_val]
    test_ids = ids[num_train + num_val:num_train + num_val + num_test]

    print("num_train", num_train)
    print("num_val", num_val)
    print("num_test", num_test)

    alpaca_prop_train_filename = os.path.join(
        config.output_dir, "alpaca_prop_train.json"
    )
    if not os.path.exists(alpaca_prop_train_filename):
        m_train = make_alpaca_json(
            dataset=dat,
            jids=train_ids,
            config=config,
        )
        dumpjson(data=m_train, filename=alpaca_prop_train_filename)
    else:
        print(alpaca_prop_train_filename, " exists")
        m_train = loadjson(alpaca_prop_train_filename)
    print("Sample:\n", m_train[0])

    alpaca_prop_val_filename = os.path.join(
        config.output_dir, "alpaca_prop_val.json"
    )
    if not os.path.exists(alpaca_prop_val_filename):
        m_val = make_alpaca_json(
            dataset=dat,
            jids=val_ids,
            config=config,
            include_jid=True,
        )
        dumpjson(data=m_val, filename=alpaca_prop_val_filename)
    else:
        print(alpaca_prop_val_filename, "exists")
        m_val = loadjson(alpaca_prop_val_filename)

    alpaca_prop_test_filename = os.path.join(
        config.output_dir, "alpaca_prop_test.json"
    )
    if not os.path.exists(alpaca_prop_test_filename):
        m_test = make_alpaca_json(
            dataset=dat,
            jids=test_ids,
            config=config,
            include_jid=True,
        )
        dumpjson(data=m_test, filename=alpaca_prop_test_filename)
    else:
        print(alpaca_prop_test_filename, "exists")
        m_test = loadjson(alpaca_prop_test_filename)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    if not isinstance(model, PeftModel):
        print("Not Peft model")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=config.lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    EOS_TOKEN = tokenizer.eos_token

    train_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_train_filename,
        split="train",
    )
    val_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_val_filename,
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=alpaca_prop_test_filename,
        split="train",
    )

    formatting_prompts_func_with_prompt = partial(
        formatting_prompts_func, alpaca_prompt=config.alpaca_prompt
    )

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    train_dataset = train_dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
        num_proc=config.dataset_num_proc
    )
    val_dataset = val_dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
        num_proc=config.dataset_num_proc
    )
    test_dataset = test_dataset.map(
        formatting_prompts_func_with_prompt,
        batched=True,
        num_proc=config.dataset_num_proc
    )

    lengths = [
        len(tokenizer(example["text"], truncation=False)["input_ids"])
        for example in val_dataset
    ]
    if lengths:
        max_seq_length = max(lengths)
        print(f"🧠 Suggested max_seq_length based on dataset: {max_seq_length}")

    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=config.dataset_num_proc)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, num_proc=config.dataset_num_proc)
    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )
    tokenized_val.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            overwrite_output_dir=True,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            output_dir=config.output_dir,
            optim=config.optim,
            seed=config.seed_val,
            num_train_epochs=config.num_epochs,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            evaluation_strategy=config.evaluation_strategy,
            eval_steps=config.eval_steps,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
            save_total_limit=config.save_total_limit,
            report_to="none",
        ),
    )

    if callback_samples > 0:
        callback = ExampleTrainerCallback(
            some_tokenized_dataset=tokenized_val,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            callback_samples=callback_samples,
        )
        trainer.add_callback(callback)
    gpu_usage = PrintGPUUsageCallback()
    trainer.add_callback(gpu_usage)
    trainer_stats = trainer.train()
    trainer.save_model(config.model_save_path)

    model = trainer.model
    FastLanguageModel.for_inference(model)

    evaluate(
        test_set=m_test,
        model=model,
        tokenizer=tokenizer,
        csv_out=config.csv_out,
        config=config,
    )
    t2 = time.time()
    print("Time taken:", t2 - t1)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main(config_file=args.config_name)
