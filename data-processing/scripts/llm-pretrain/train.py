import argparse
import json
import os
from datetime import datetime
import datasets
from datasets import  Features, Value ,Sequence
import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, logging, AutoModelForCausalLM, TrainerState
)
from trl import SFTTrainer, SFTConfig, get_kbit_device_map
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator

def main(args_pars):
    # Quantization config (QLoRA)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float32,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, inference_mode=False, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
    )

    # Directory setup
    run_path = f'{args_pars.root}/{args_pars.rname}'
    output_dir = f'{run_path}/checkpoint/'
    save_run = f'{run_path}/model/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_run, exist_ok=True)

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args_pars.model,
        torch_dtype=torch.float16,
        quantization_config=nf4_config,
        attn_implementation="sdpa",
        device_map=get_kbit_device_map()
    )
    tokenizer = AutoTokenizer.from_pretrained(args_pars.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data (expects .jsonl, one log per line)
    data_files = {}
    if os.path.exists(args_pars.train_file):
        data_files["train"] = args_pars.train_file
    if args_pars.valid_file and os.path.exists(args_pars.valid_file):
        data_files["validation"] = args_pars.valid_file

    with open("/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/features.json") as f:
        feat_dict = json.load(f)

    features_dict = {k: Value(v) for k, v in feat_dict.items()}
    # Add 'tags' if missing
    features_dict['tags'] = Sequence(Value("string"))

    features = Features(features_dict)
    
    dataset = datasets.load_dataset("json", data_files=data_files, field=None, features=features)
    train_data = dataset["train"]
    eval_data = dataset["validation"] if "validation" in dataset else None

    # Debugging info
    import psutil
    print("==== DEBUG: Dataset loaded ====")
    print("Train dataset type:", type(train_data))
    print("Train dataset length:", len(train_data))
    print("Train dataset columns:", train_data.column_names)
    print("RAM used (GB):", psutil.Process(os.getpid()).memory_info().rss / 1e9)
    print("First train_data row (repr):", repr(train_data[0]))
    print("First train_data row (json):", json.dumps(train_data[0], ensure_ascii=False))
    print("===============================")


    def preprocess_function(examples):
        import json
        # On prépare une liste vide pour stocker chaque log sous forme de string
        logs = []
        # On calcule combien d'exemples (lignes) il y a dans ce batch
        num_examples = len(next(iter(examples.values())))
        # Pour chaque ligne du batch :
        for i in range(num_examples):
            # On construit un dictionnaire {colonne: valeur} pour la i-ème ligne
            row = {k: v[i] for k, v in examples.items()}
            # On convertit ce dictionnaire en string JSON (toutes les infos du log sont là)
            logs.append(json.dumps(row, ensure_ascii=False))
        # On applique le tokenizer sur chaque string (troncature/padding si besoin)
        return tokenizer(
            logs,
            truncation=True,
            padding="max_length",
            max_length=args_pars.context
        )


    # Tokenize
    try:
        train_data = train_data.map(
            preprocess_function, batched=True, remove_columns=train_data.column_names
        )
    except Exception as e:
        print("Exception during train_data.map:", e)
        import traceback; traceback.print_exc()
        exit(1)

    if eval_data:
        try:
            eval_data = eval_data.map(
                preprocess_function, batched=True, remove_columns=eval_data.column_names
            )
        except Exception as e:
            print("Exception during eval_data.map:", e)
            import traceback; traceback.print_exc()
            exit(1)
  # ---- DEBUG APRÈS TOKENIZATION ----
    print("==== DEBUG: Dataset after tokenization ====")
    if hasattr(train_data, 'shape'):
        print("Shape after map:", train_data.shape)
    print("Train dataset length:", len(train_data))
    print("Train dataset columns:", train_data.column_names)
    try:
        print("First row after tokenization:", train_data[0])
    except Exception as e:
        print("Could not print first tokenized row:", e)
    print("RAM used (GB):", psutil.Process(os.getpid()).memory_info().rss / 1e9)
    print("===============================")

    # Training args
    train_args = SFTConfig(
        do_train=True,
        per_device_train_batch_size=args_pars.batch,
        per_device_eval_batch_size=args_pars.batch,
        gradient_accumulation_steps=args_pars.grad,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="steps",
        eval_steps=100,
        fp16_full_eval=True,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        warmup_steps=50,
        num_train_epochs=args_pars.epochs,
        report_to="wandb",
        learning_rate=args_pars.lr,
        output_dir=output_dir,
        max_seq_length=args_pars.context,
        dataset_text_field="text",
        disable_tqdm=True,
        run_name=args_pars.rname,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=train_args,
        tokenizer=tokenizer,
        formatting_func=None,
    )

    trainer.accelerator.print(
        f"\nTraining model : {args_pars.model}\n"
        f"\trun : {args_pars.rname},\n"
        f"\tdataset : {args_pars.train_file}\n"
        f"\tlearning_rate : {train_args.learning_rate},\n"
        f"\tcontext : {args_pars.context},\n"
        f"\tbatch : {args_pars.batch},\n"
        f"\toutput_dir : {output_dir}\n"
        f"\tstart_time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # --- Checkpoint logic ---
    resume_from = None
    if args_pars.checkpoint:
        resume_from = get_last_checkpoint(output_dir)
        if resume_from is not None:
            trainer.accelerator.print(f"Resuming from checkpoint: {resume_from}")
        else:
            trainer.accelerator.print("No checkpoint found, starting from scratch.")

    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(save_run)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True, dest="rname")
    parser.add_argument("--train-file", type=str, required=True, help="Path to shuffled train.jsonl")
    parser.add_argument("--valid-file", type=str, default=None, help="Optional valid.jsonl")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--root", type=str, default="./run")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, help="Resume from last checkpoint if available")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    return parser

if __name__ == "__main__":
    args_pars = setup_parser().parse_args()
    logging.disable_progress_bar()
    datasets.disable_progress_bars()

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        'LogAP-LLM',
        init_kwargs={
            "wandb": {
                "mode": "offline",
                'dir': f'{args_pars.root}/{args_pars.rname}/',
                "resume": "auto" if args_pars.checkpoint else None,
                "name": args_pars.rname
            }
        }
    )
    main(args_pars)
    accelerator.end_training()
