from atomgpt.inverse_models.loader import FastLanguageModel
from atomgpt.inverse_models.inverse_models import TrainingPropConfig
from jarvis.db.jsonutils import loadjson, dumpjson
import os
import pprint
from atomgpt.inverse_models.utils import gen_atoms, main_spectra, load_exp_file
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from ase.optimize.fire import FIRE
from ase.constraints import ExpCellFilter
import time
from jarvis.core.atoms import ase_to_atoms
import re  # CHANGE: used to detect formula-like filenames

parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer"
    + " Inverse Model Predictor."
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Name of the output directory",
)
parser.add_argument(
    "--pred_csv",
    default="pred_list_inverse.csv",
    help="CSV file for prediction list.",
)
parser.add_argument(
    "--intvl",
    default="0.3",
    help="XRD 2 theta bin",
)
parser.add_argument(
    "--relax",
    default="True",
    help="Relax cell or not",
)
parser.add_argument(
    "--background_subs",
    default="False",
    help="Perform background subtraction",
)
parser.add_argument(
    "--model_name",
    default=None,
    help="Name or path of model if not using config.json",
)
parser.add_argument(
    "--dat_path",
    default=None,
    help="Spectra .dat path with X and Y ",
)
parser.add_argument(
    "--formula",
    default=None,
    help="Chemical formula ",
)
parser.add_argument(
    "--prop_val",
    default=None,
    help="Property values ",
)
parser.add_argument(
    "--config_path",
    default=None,
    help="Chemical formula ",
)


def relax_atoms(
    atoms=None,
    fmax=0.05,
    nsteps=150,
    constant_volume=False,
):
    from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

    calculator = AlignnAtomwiseCalculator(path=default_path(), device="cpu")
    t1 = time.time()
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator

    ase_atoms = ExpCellFilter(ase_atoms, constant_volume=constant_volume)
    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=nsteps)
    en = ase_atoms.atoms.get_potential_energy()
    final_atoms = ase_to_atoms(ase_atoms.atoms)
    t2 = time.time()
    return final_atoms


def predict(
    output_dir=None,
    config_path=None,
    pred_csv="pred_list_inverse.csv",
    fname="out_inv.json",
    device="cuda",
    intvl=0.3,
    tol=0.1,
    relax=False,
    model_name=None,
    dat_path=None,
    background_subs=False,
    filename="Q4_K_M.gguf",
    formula=None,
    prop_val=None,
    dtype=None,
    max_seq_length=1058,
    load_in_4bit=None,  # temp_config["load_in_4bit"]
    verbose=True,  # temp_config["load_in_4bit"]
):
    # Track whether the user provided a formula via CLI
    user_provided_formula = formula is not None

    print("config_path", config_path)
    if output_dir is not None:
        config_name = os.path.join(output_dir, "config.json")
        parent = Path(output_dir).parent
        if not os.path.exists(config_name):
            config_name = os.path.join(parent, "config.json")
        adapter = os.path.join(output_dir, "adapter_config.json")
        if os.path.exists(adapter):
            model_name = output_dir
    if config_path is not None:
        config_name = config_path
        if verbose:
            print("config used", config_name)
        temp_config = loadjson(config_name)
        if verbose:
            print("config used", temp_config)
        temp_config = TrainingPropConfig(**temp_config).dict()
        max_seq_length = temp_config["max_seq_length"]
        dtype = temp_config["dtype"]
    # temp_config = TrainingPropConfig().dict()
    temp_config = TrainingPropConfig(**temp_config).dict()
    if verbose:
        pprint.pprint(temp_config)
    if model_name is None:
        model_name = temp_config["model_name"]
    # output_dir = temp_config["output_dir"]
    if load_in_4bit is None:
        load_in_4bit = temp_config["load_in_4bit"]

    if verbose:
        print("Model used:", model_name)
        print("config used:", config_path)
        print("formula:", formula)
        print("dat_path:", dat_path)

    model = None
    tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, gguf_file=filename
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, gguf_file=filename
        )
        pass

    atoms_arr = []
    lines = []

    # If formula isn't given but dat_path is, run single-file inference (no pred_csv needed)
    if formula is None:
        if dat_path is not None:
            lines = [dat_path]
        else:
            f = open(pred_csv, "r")
            lines = f.read().splitlines()
            f.close()
    else:
        if dat_path is not None:
            lines = [dat_path]
        lines = [formula]

    mem = []

    # CHANGE: simple heuristic for "formula-like" strings (e.g., LaB6, SiO2)
    formula_like_re = re.compile(r"^([A-Z][a-z]?\d*)+$")

    for i in lines:
        prompt = i
        if ".dat" in i or dat_path is not None:
            if dat_path is None:
                parent = Path(pred_csv).parent
                fname_csv = os.path.join(parent, i)
            else:
                fname_csv = dat_path

            # Determine whether the DAT filename itself looks like a chemical formula
            fname_stem = Path(fname_csv).stem
            filename_looks_like_formula = bool(formula_like_re.fullmatch(fname_stem))

            _formula, x, y = load_exp_file(
                filename=fname_csv,
                intvl=intvl,
                tol=tol,
                formula=formula,
                background_subs=background_subs,
            )
            y_new_str = y
            try:
                if ".dat" in i:
                    formula = str(_formula.split("/")[-1].split(".dat")[0])
            except Exception:
                pass

            # CHANGE: inject formula sentence only if:
            #   - user provided --formula, OR
            #   - DAT filename itself is a formula (e.g., LaB6.dat)
            if user_provided_formula and formula is not None:
                prompt = (
                    "The chemical formula is "
                    + formula
                    + " The "
                    + temp_config["prop"]
                    + " is "
                    + y_new_str
                    + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
                )
            elif (not user_provided_formula) and filename_looks_like_formula:
                prompt = (
                    "The chemical formula is "
                    + fname_stem
                    + " The "
                    + temp_config["prop"]
                    + " is "
                    + y_new_str
                    + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
                )
            else:
                prompt = (
                    "The "
                    + temp_config["prop"]
                    + " is "
                    + y_new_str
                    + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
                )
        else:
            if user_provided_formula and formula is not None:
                prompt = (
                    "The chemical formula is "
                    + formula
                    + " The "
                    + temp_config["prop"]
                    + " is "
                    + str(prop_val)
                    + ". Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
                )

        if verbose:
            print("prompt here", prompt.replace("\n", ","))

        gen_mat = gen_atoms(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            alpaca_prompt=temp_config["alpaca_prompt"],
            instruction=temp_config["instruction"],
            device=device,
        )

        if gen_mat is None:
            print(
                "The structure returned by gen_mat() is not a valid crystal structure."
            )
            info = {}
            info["prompt"] = prompt
            info["error"] = "Invalid structure returned by AtomGPT (None)."
            mem.append(info)
            continue

        if verbose:
            print("gen atoms", gen_mat)
            print("gen atoms spacegroup", gen_mat.spacegroup())
            print("intvl", intvl)
        if relax:
            gen_mat = relax_atoms(atoms=gen_mat)
            if verbose:
                print("gen atoms relax", gen_mat, gen_mat.spacegroup())
        atoms_arr.append(gen_mat.to_dict())
        info = {}
        info["prompt"] = prompt
        info["atoms"] = gen_mat.to_dict()
        mem.append(info)

    dumpjson(data=mem, filename=fname)
    return model, tokenizer, temp_config


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    print("args.config_path", args.config_path)
    predict(
        output_dir=args.output_dir,
        pred_csv=args.pred_csv,
        intvl=float(args.intvl),
        model_name=args.model_name,
        dat_path=args.dat_path,
        formula=args.formula,
        config_path=args.config_path,
        prop_val=args.prop_val,
        background_subs=args.background_subs,
    )

