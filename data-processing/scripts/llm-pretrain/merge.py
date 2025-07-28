import argparse
import torch
from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", action='store', type=str, required=True)
parser.add_argument("--out", action='store', type=str, required=True)
args_pars = parser.parse_args()

def main():
    config = PeftConfig.from_pretrained(args_pars.model)

    print("Loading base model :", config.base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", local_files_only=True, torch_dtype=torch.float16)

    print("Loading PEFT model :", args_pars.model)
    model = PeftModel.from_pretrained(base_model, args_pars.model, device_map="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args_pars.model)

    print("Merge models ...")
    merged = model.merge_and_unload()

    print("Save merged model :", args_pars.out, f"({merged.dtype})")
    merged.save_pretrained(args_pars.out)
    tokenizer.save_pretrained(args_pars.out)

if __name__ == "__main__":
    main()