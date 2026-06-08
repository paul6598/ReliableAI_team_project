import json
import csv
import argparse
from collections import defaultdict

import jiwer


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def compute_cer(ref, hyp):
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if len(ref) == 0:
        return None

    return jiwer.process_characters(ref, hyp).cer


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def degradation(clean, adv):
    if clean is None or adv is None:
        return None, None, None

    delta = adv - clean
    rel = None if clean == 0 else (delta / clean) * 100
    headroom = None if clean >= 1 else (delta / (1 - clean)) * 100

    return delta, rel, headroom


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def load_grouped(path):
    grouped = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[row["lang_tag"]].append(row)

    return grouped


def evaluate(jsonl_path):
    grouped = load_grouped(jsonl_path)
    results = []

    print("=== CER ONLY EVALUATION ===")
    print("lang\tN\tclean_CER\tadv_CER\tCER_delta\tCER_rel_%\tCER_headroom_%")

    for lang in sorted(grouped):
        rows = grouped[lang]

        clean_cers = []
        adv_cers = []

        for row in rows:
            ref = row["ground_truth"]
            clean = row["clean_pred"]
            adv = row["adv_pred"]

            clean_cers.append(compute_cer(ref, clean))
            adv_cers.append(compute_cer(ref, adv))

        clean_cer = safe_mean(clean_cers)
        adv_cer = safe_mean(adv_cers)
        cer_delta, cer_rel, cer_headroom = degradation(clean_cer, adv_cer)

        result = {
            "language": lang,
            "num_samples": len(rows),
            "clean_CER": clean_cer,
            "adv_CER": adv_cer,
            "CER_delta": cer_delta,
            "CER_relative_percent": cer_rel,
            "CER_headroom_percent": cer_headroom,
        }

        results.append(result)

        print(
            f"{lang}\t{len(rows)}\t"
            f"{fmt(clean_cer)}\t{fmt(adv_cer)}\t"
            f"{fmt(cer_delta)}\t{fmt(cer_rel)}\t{fmt(cer_headroom)}"
        )

    return results


def save_csv(results, output_csv):
    fields = [
        "language",
        "num_samples",
        "clean_CER",
        "adv_CER",
        "CER_delta",
        "CER_relative_percent",
        "CER_headroom_percent",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("Saved CSV:", output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    args = parser.parse_args()

    results = evaluate(args.jsonl)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
