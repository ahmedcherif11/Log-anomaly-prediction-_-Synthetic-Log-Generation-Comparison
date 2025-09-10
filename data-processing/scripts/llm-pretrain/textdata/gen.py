import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_from_disk, disable_progress_bars
from transformers import AutoTokenizer, AutoModelForCausalLM

RESPONSE_MARKER = "### Response:\n"

def extract_response(text: str) -> str:
    return text.split(RESPONSE_MARKER, 1)[-1].strip() if RESPONSE_MARKER in text else text.strip()

def main(args):
    
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    device_map = "auto" if cuda_ok else {"": "cpu"}
    print(f"Using device map: {device_map}")

    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if cuda_ok else torch.float32,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    disable_progress_bars()
    print(f"Loading dataset from {args.dataset}")
    ds = load_from_disk(args.dataset)
    # Use "test" split if present, else first available split
    if "test" in ds:
        samples = ds["test"]
    else:
        samples = ds[list(ds.keys())[0]]

    results = []
    for ex in tqdm(samples):
        prompt = ex["prompt"]
        technique = ex.get("technique", "")
        tail = "Output exactly one Windows event log in JSON. No explanations. One object only."
        gen_prompt = f"### Prompt:\n{prompt.strip()}\n\n{tail}\n\n{RESPONSE_MARKER}"
        inputs = tokenizer(gen_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.45,
                no_repeat_ngram_size=10,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        gen_text = extract_response(decoded)
        results.append({"technique": technique, "prompt": prompt, "generated": gen_text})

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} generations to {args.output_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, help="Input HuggingFace dataset folder")
    ap.add_argument("--output_jsonl", type=str, required=True, help="Output generations JSONL file")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    args = ap.parse_args()
    main(args)