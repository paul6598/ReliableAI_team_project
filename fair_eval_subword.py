import json
import argparse
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def edit_distance(a, b):
    """
    list token sequence edit distance.
    a: reference token ids
    b: hypothesis token ids
    """
    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            cost = 0 if ai == bj else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


def subword_error_rate(ref_text, hyp_text, tokenizer):
    """
    Subword Error Rate.
    같은 multilingual tokenizer로 ref/hyp를 tokenize한 뒤,
    token id sequence edit distance / reference token length 계산.
    """
    ref_text = normalize_text(ref_text)
    hyp_text = normalize_text(hyp_text)

    ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
    hyp_ids = tokenizer.encode(hyp_text, add_special_tokens=False)

    if len(ref_ids) == 0:
        return None

    dist = edit_distance(ref_ids, hyp_ids)
    return dist / len(ref_ids)


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def degradation(clean, adv):
    if clean is None or adv is None:
        return None, None, None

    delta = adv - clean
    relative_percent = None if clean == 0 else (delta / clean) * 100
    headroom_percent = None if clean >= 1 else (delta / (1 - clean)) * 100

    return delta, relative_percent, headroom_percent


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def evaluate_jsonl(path, tokenizer_name):
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rows = load_jsonl(path)

    grouped = defaultdict(list)

    for row in rows:
        lang = row["lang_tag"]
        grouped[lang].append({
            "ref": row["ground_truth"],
            "clean": row["clean_pred"],
            "adv": row["adv_pred"],
        })

    results = {}

    for lang, items in sorted(grouped.items()):
        print(f"Evaluating language: {lang}, N={len(items)}")

        clean_swers = []
        adv_swers = []

        for idx, item in enumerate(items, start=1):
            ref = item["ref"]
            clean = item["clean"]
            adv = item["adv"]

            clean_swers.append(subword_error_rate(ref, clean, tokenizer))
            adv_swers.append(subword_error_rate(ref, adv, tokenizer))

            if idx % 200 == 0:
                print(f"  {lang}: {idx}/{len(items)}")

        clean_swer = safe_mean(clean_swers)
        adv_swer = safe_mean(adv_swers)

        swer_delta, swer_rel, swer_headroom = degradation(clean_swer, adv_swer)

        results[lang] = {
            "num_samples": len(items),
            "clean_SWER": clean_swer,
            "adv_SWER": adv_swer,
            "SWER_delta": swer_delta,
            "SWER_relative_percent": swer_rel,
            "SWER_headroom_percent": swer_headroom,
        }

    return results


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def print_table(results):
    print("\n=== SUBWORD TOKEN EVALUATION RESULTS ===")
    print(
        "lang\tN\tclean_SWER\tadv_SWER\tSWER_delta\tSWER_rel_%\tSWER_headroom_%"
    )

    for lang, r in results.items():
        print(
            f"{lang}\t"
            f"{r['num_samples']}\t"
            f"{fmt(r['clean_SWER'])}\t"
            f"{fmt(r['adv_SWER'])}\t"
            f"{fmt(r['SWER_delta'])}\t"
            f"{fmt(r['SWER_relative_percent'])}\t"
            f"{fmt(r['SWER_headroom_percent'])}"
        )


def save_csv(results, output_path):
    import csv

    fields = [
        "language",
        "num_samples",
        "clean_SWER",
        "adv_SWER",
        "SWER_delta",
        "SWER_relative_percent",
        "SWER_headroom_percent",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for lang, r in results.items():
            row = {"language": lang}
            row.update(r)
            writer.writerow(row)

    print("\nSaved CSV:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        default="./data/attack_results/all_results.jsonl",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./data/attack_results/fair_eval_subword_summary.csv",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="xlm-roberta-base",
    )

    args = parser.parse_args()

    results = evaluate_jsonl(args.jsonl, args.tokenizer)
    print_table(results)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
