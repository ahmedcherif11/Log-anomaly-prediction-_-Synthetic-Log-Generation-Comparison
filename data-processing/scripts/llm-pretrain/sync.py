
import credentials
import os
import datasets
import torch
import argparse
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", action="store", type=str, help="path to data", default="../data/temp/", dest="data_path")
parser.add_argument("--save-model", action="store", type=bool, help="save model in huggingface", default=False, dest="save_model")
parser.add_argument("--save-dataset", action="store", type=bool, help="save dataset in huggingface", default=False, dest="save_dataset")
parser.add_argument("--wandb", action="store", type=bool, help="save run in wandb", default=True)
args_pars = parser.parse_args()


def main():
    if args_pars.wandb:
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_API_KEY"] = credentials.wandb_key

    if args_pars.save_dataset or args_pars.save_model:
        login(token=credentials.hf_key)

        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

        if args_pars.save_model:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", quantization_config=nf4_config, device_map=device_map)
            # model.load_adapter("./data/model")
            model = AutoModelForCausalLM.from_pretrained(f"{args_pars.data_path}run/model",
                                                         quantization_config=nf4_config,
                                                         device_map=device_map,
                                                         local_files_only=True)

            tokenizer = AutoTokenizer.from_pretrained(f"{args_pars.data_path}run/model", local_files_only=True)
            model.push_to_hub('Blackam09/raw-logs')
            tokenizer.push_to_hub('Blackam09/raw-logs')

        if args_pars.save_dataset:
            data = datasets.load_from_disk(f"{args_pars.data_path}tokenized/loghubQwe_tok")
            data.push_to_hub('Blackam09/raw-logs')


if __name__ == "__main__":
    main()
