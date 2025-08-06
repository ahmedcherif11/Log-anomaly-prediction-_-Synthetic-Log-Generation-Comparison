import argparse
import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

# Optional metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def parse_log(log):
    """Try to parse log as JSON dict. Return None if not possible."""
    try:
        return json.loads(log)
    except Exception:
        return None

def jaccard_for_dicts(d1, d2):
    """Jaccard over items (field, value) pairs for JSON logs."""
    if not d1 or not d2:
        return 0.0
    set1, set2 = set(d1.items()), set(d2.items())
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)

def ngram_overlap(s1, s2, n=2):
    """Jaccard similarity over n-grams (default bigram) for text logs."""
    def ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    t1, t2 = s1.split(), s2.split()
    ngrams1, ngrams2 = ngrams(t1, n), ngrams(t2, n)
    if not ngrams1 or not ngrams2:
        return 0.0
    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)

def main(args):
    # Load model & tokenizer
    print("Loading model from", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # Load data (can be a Hugging Face dataset dir or a JSONL file with prompts)
    print(f"Loading dataset from {args.dataset}")
    try:
        data = datasets.load_from_disk(args.dataset)
        # Try 'test' split, else fall back to full dataset
        if "test" in data:
            samples = data["test"]
        else:
            samples = data[list(data.keys())[0]]
    except Exception:
        # Try to load as plain JSONL with list of dicts with 'prompt' key
        with open(args.dataset, "r") as f:
            samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples.")

    has_reference = "response" in samples[0]  # assume all-or-nothing
    results = []
    bleu_scores, jaccard_scores, ngram_scores = [], [], []

    for example in tqdm(samples):
        prompt = example["prompt"]
        gt_response = example.get("response", "")
        # Model generation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        record = {"prompt": prompt, "generated": generated}
        if has_reference:
            record["reference"] = gt_response
            # --- METRICS ---
            # 1. BLEU
            bleu = sentence_bleu(
                [gt_response.split()],
                generated.split(),
                smoothing_function=SmoothingFunction().method1,
            )
            bleu_scores.append(bleu)
            # 2. Jaccard (field/value pairs for JSON)
            dict_gt = parse_log(gt_response)
            dict_gen = parse_log(generated)
            jaccard = jaccard_for_dicts(dict_gt, dict_gen)
            jaccard_scores.append(jaccard)
            # 3. Bigram n-gram Jaccard
            ngram_jaccard = ngram_overlap(gt_response, generated, n=2)
            ngram_scores.append(ngram_jaccard)
            record.update({"bleu": bleu, "jaccard": jaccard, "ngram_jaccard": ngram_jaccard})

        results.append(record)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved all generations and metrics to {results_path}")

    if has_reference:
        print("\n========== Evaluation Metrics (on test set) ==========")
        print(f"Average BLEU score:          {sum(bleu_scores)/len(bleu_scores):.4f}")
        print(f"Average Jaccard (fields):    {sum(jaccard_scores)/len(jaccard_scores):.4f}")
        print(f"Average Bigram Jaccard:      {sum(ngram_scores)/len(ngram_scores):.4f}")
    else:
        print("\n[Info] No ground-truth reference responses found. Only synthetic log generation performed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model checkpoint directory (can be merged or adapter model)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset (HuggingFace disk format or JSONL file)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()
    main(args)
