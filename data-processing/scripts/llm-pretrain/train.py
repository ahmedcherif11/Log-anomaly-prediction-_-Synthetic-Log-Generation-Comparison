import argparse
import json
import os
from datetime import datetime
import datasets
from datasets import Features, Value, Sequence
import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, logging, AutoModelForCausalLM
)
from trl import SFTTrainer, SFTConfig, get_kbit_device_map
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator

def main(args_pars, run_path, output_dir, save_run, accelerator):
    # --- 1. Quantization config (QLoRA) ---
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float32,
    )

    # --- 2. LoRA config ---
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1, inference_mode=False, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
    )

    # --- 3. Directory setup ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_run, exist_ok=True)

    # --- 4. Load model/tokenizer (no FSDP wrapping ici !) ---
    model = AutoModelForCausalLM.from_pretrained(
        args_pars.model,
        torch_dtype=torch.float16,
        quantization_config=nf4_config,
        attn_implementation="sdpa",
        device_map=None,         # ← NE PAS utiliser get_kbit_device_map ici ! (Accelerate gère la distrib)
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args_pars.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # --- 5. PREPARE MODEL FOR DISTRIBUTED/FSDP via Accelerate ---
    # Cette ligne est cruciale : c’est elle qui wrappe le modèle avec FSDP, DDP, etc. selon la config.
    model = accelerator.prepare(model)

    # --- 6. DATA LOADING ---
    data_files = {}
    if os.path.exists(args_pars.train_file):
        data_files["train"] = args_pars.train_file
    if args_pars.valid_file and os.path.exists(args_pars.valid_file):
        data_files["validation"] = args_pars.valid_file

    with open("/project/def-dmouheb/cherif/Log-anomaly-prediction-_-Synthetic-Log-Generation-Comparison/data-processing/scripts/llm-pretrain/features.json") as f:
        feat_dict = json.load(f)
    features_dict = {k: Value(v) for k, v in feat_dict.items()}
    features_dict['tags'] = Sequence(Value("string"))
    features = Features(features_dict)

    dataset = datasets.load_dataset("json", data_files=data_files, field=None, features=features)
    train_data = dataset["train"]
    eval_data = dataset["validation"] if "validation" in dataset else None

    # --- 7. DEBUG: After dataset load ---
    import psutil
    accelerator.print("==== DEBUG: Dataset loaded ====")
    accelerator.print("Train dataset type:", type(train_data))
    accelerator.print("Train dataset length:", len(train_data))
    accelerator.print("RAM used (GB):", psutil.Process(os.getpid()).memory_info().rss / 1e9)
    accelerator.print("===============================")
    accelerator.print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # --- 8. TOKENIZATION ---
    def preprocess_function(examples):
        logs = []
        num_examples = len(next(iter(examples.values())))
        for i in range(num_examples):
            row = {k: v[i] for k, v in examples.items()}
            logs.append(json.dumps(row, ensure_ascii=False))
        return tokenizer(
            logs,
            truncation=True,
            padding="max_length",
            max_length=args_pars.context
        )

    try:
        train_data = train_data.map(
            preprocess_function, batched=True, remove_columns=train_data.column_names
        )
    except Exception as e:
        accelerator.print("Exception during train_data.map:", e)
        import traceback; traceback.print_exc()
        exit(1)
    if eval_data:
        try:
            eval_data = eval_data.map(
                preprocess_function, batched=True, remove_columns=eval_data.column_names
            )
        except Exception as e:
            accelerator.print("Exception during eval_data.map:", e)
            import traceback; traceback.print_exc()
            exit(1)

    accelerator.print("==== DEBUG: Dataset after tokenization ====")
    if hasattr(train_data, 'shape'):
        accelerator.print("Shape after map:", train_data.shape)
    accelerator.print("Train dataset length:", len(train_data))
    try:
        accelerator.print("First row after tokenization:", train_data[0])
    except Exception as e:
        accelerator.print("Could not print first tokenized row:", e)
    accelerator.print("RAM used (GB):", psutil.Process(os.getpid()).memory_info().rss / 1e9)
    accelerator.print("===============================")

    # --- 9. TRAINING ARGS ---
    train_args = SFTConfig(
        do_train=True,
        per_device_train_batch_size=args_pars.batch,
        per_device_eval_batch_size=args_pars.batch,
        gradient_accumulation_steps=args_pars.grad,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        lr_scheduler_type="cosine",
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

    # --- 10. SFT TRAINER ---
    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=train_args,
        processing_class=tokenizer,
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

    # --- 11. CHECKPOINT LOGIC ---
    resume_from = get_last_checkpoint(output_dir) if args_pars.checkpoint else None
    if args_pars.checkpoint and resume_from:
        trainer.accelerator.print(f"Resuming from checkpoint: {resume_from}")
    elif args_pars.checkpoint:
        trainer.accelerator.print("No checkpoint found, starting from scratch.")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(save_run)
    tokenizer.save_pretrained(save_run)

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
    run_path = f'{args_pars.root}/{args_pars.rname}'
    output_dir = f'{run_path}/checkpoint/'
    save_run = f'{run_path}/model/'
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        'LogAP-LLM',
        init_kwargs={
            "wandb": {
                "mode": "offline",
                'dir': f'{run_path}/',
                "resume": "auto" if args_pars.checkpoint else None,
                "name": args_pars.rname
            }
        }
    )
    logging.disable_progress_bar()
    datasets.disable_progress_bars()
    main(args_pars, run_path, output_dir, save_run, accelerator)
    accelerator.end_training()
    