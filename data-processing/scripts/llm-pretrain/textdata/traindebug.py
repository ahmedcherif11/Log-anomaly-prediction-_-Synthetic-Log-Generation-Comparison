import argparse
import os
from datetime import datetime
import sys
import traceback

import datasets
import torch
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig, logging, AutoModelForCausalLM, TrainerState
from transformers.trainer import TRAINER_STATE_NAME
from trl import SFTTrainer, SFTConfig, get_kbit_device_map
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator

# ------- Early Distributed Debug Logging --------
def early_debug_log(msg):
    try:
        rank = int(os.environ.get("RANK", 0))
    except Exception:
        rank = "NA"
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    except Exception:
        local_rank = "NA"
    print(f"[RANK={rank} LOCAL_RANK={local_rank} PID={os.getpid()}] {msg}", flush=True)

def main():
    try:
        early_debug_log("START: main() entry")
        # Quantization configuration
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage=torch.float32,
        )

        early_debug_log("Instantiating LoRA config")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            inference_mode=False,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )

        early_debug_log("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args_pars.model,
            torch_dtype=torch.float32,
            quantization_config=nf4_config,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map=get_kbit_device_map(),
        )
        early_debug_log("Base model loaded.")

        early_debug_log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args_pars.model, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        early_debug_log("Tokenizer loaded.")

        early_debug_log(f"Loading dataset from {args_pars.dataset} ...")
        data = datasets.load_from_disk(args_pars.dataset)
        early_debug_log("Dataset loaded.")

        import psutil
        early_debug_log(f"Dataset stats: type={type(data['train'])}, len={len(data['train'])}, RAM used (GB): {psutil.Process(os.getpid()).memory_info().rss / 1e9}")
        print(torch.cuda.memory_summary(device=None, abbreviated=False), flush=True)

        early_debug_log("Defining training configuration (SFTConfig)...")
        args = SFTConfig(
            do_train=True,
            per_device_train_batch_size=args_pars.batch,
            per_device_eval_batch_size=args_pars.batch,
            gradient_accumulation_steps=args_pars.grad,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            eval_strategy="steps",
            eval_steps=50,
            fp16_full_eval=True,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=5,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            warmup_steps=10,
            num_train_epochs=1,
            report_to="wandb",
            learning_rate=5e-4,
            output_dir=output_dir,
            max_seq_length=args_pars.context,
            dataset_text_field='text',
            disable_tqdm=True,
            run_name=args_pars.rname,
            packing=True,
        )

        early_debug_log("Creating SFTTrainer...")
        trainer = SFTTrainer(
            model=base_model,
            peft_config=peft_config,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            args=args,
            processing_class=tokenizer,
        )
        early_debug_log("SFTTrainer created.")

        # Checkpoints logic
        if args_pars.is_save_checkpoint:
            early_debug_log("Checking for checkpoint resume...")
            if isinstance(output_dir, bool) and output_dir:
                resume_from_checkpoint = get_last_checkpoint(output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(f"No valid checkpoint found in output directory ({output_dir})")

                state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                trainer._load_from_checkpoint(resume_from_checkpoint=state.best_model_checkpoint)
        else:
            early_debug_log("Trainer state (pre-training):")
            trainer.accelerator.print(f"{trainer.model}")
            if getattr(trainer.accelerator.state, "fsdp_plugin", None):
                from peft.utils.other import fsdp_auto_wrap_policy
                fsdp_plugin = trainer.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

            trainer.accelerator.print(f"\nTraining model : {args_pars.model} on PEFT\n"
                f'\t"run" : "{args_pars.rname}",\n'
                f'\t"output_saved" : "{save_run}",\n'
                f'\t"dataset" : "{args_pars.dataset}",\n'
                f'\t\t"learning_rate" : "{args.learning_rate}",\n'
                f'\t\t"context" : "{args_pars.context}",\n'
                f'\t\t"batch" : "{args_pars.batch}",\n'
                f'\t\t"is_checkpoint" : "{args_pars.checkpoint}",\n'
                f'\t\t"checkpoint_path" : "{output_dir}",\n'
                f'\t"start_time" : "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"\n\n')

            early_debug_log("Starting trainer.train() ...")
            trainer.train(resume_from_checkpoint=args_pars.checkpoint)
            early_debug_log("trainer.train() finished.")

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        early_debug_log("Saving model ...")
        trainer.save_model(save_run)
        early_debug_log("Model saved.")

    except Exception as e:
        # This ensures crash tracebacks are printed on *all* ranks!
        early_debug_log("EXCEPTION in main()")
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # Optionally write to a log file for further investigation
        with open(f"debug_crash_rank{os.environ.get('RANK','0')}.log", "w") as f:
            f.write(traceback.format_exc())
        raise e

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action='store', type=str, required=True)
    parser.add_argument("--run-name", action='store', type=str, required=True, dest='rname')
    parser.add_argument("--dataset", action="store", type=str, help="dataset path")
    parser.add_argument("--batch", action="store", type=int, help="batch size", default=1)
    parser.add_argument("--grad", action="store", type=int, help="gradient accumulation step", default=4)
    parser.add_argument("--context", action="store", type=int, help="context size (input)", default=2048)
    parser.add_argument("--root", action="store", type=str, help="root path (default ./run)", default="./run")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, help="Resume from checkpoint or not")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, help="Whether to save fsdp peft training from existing checkpoint", dest="is_save_checkpoint")
    return parser

if __name__ == "__main__":
    early_debug_log("Script startup: __main__")
    args_pars = setup_parser().parse_args()

    run_path = f'{args_pars.root}/{args_pars.rname}'
    save_run = f'{run_path}/model/'
    output_dir = f'{run_path}/checkpoint/'
    run_name = "exp_" + args_pars.rname

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        'LogAP-LLM',
        init_kwargs={
            "wandb": {
                "mode" : "offline",
                'dir': f'{run_path}/',
                "resume":"auto" if args_pars.checkpoint else None,
                "name":run_name
            }
        }
    )

    logging.disable_progress_bar()
    datasets.disable_progress_bars()

    main()

    accelerator.end_training()
