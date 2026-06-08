import json
import argparse
from pathlib import Path
from collections import defaultdict
import jiwer


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def cer(ref: str, hyp: str):
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if len(ref) == 0:
        return None

    return jiwer.process_characters(ref, hyp).cer


def wer(ref: str, hyp: str):
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if len(ref.split()) == 0:
        return None

    return jiwer.process_words(ref, hyp).wer


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


def detect_field(row, candidates):
    for key in candidates:
        if key in row:
            return key
    return None


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def evaluate_jsonl(path):
    rows = load_jsonl(path)

    if not rows:
        raise ValueError("No rows found.")

    sample = rows[0]

    lang_key = detect_field(sample, [
        "lang", "lang_tag", "language", "language_code"
    ])
    ref_key = detect_field(sample, [
        "gt", "ground_truth", "reference", "ref_text",
        "text", "transcription", "raw_transcription"
    ])
    clean_key = detect_field(sample, [
        "clean_pred", "original_pred", "orig_pred",
        "prediction_clean", "pred_clean", "hyp_clean",
        "original_prediction", "clean_prediction"
    ])
    adv_key = detect_field(sample, [
        "adv_pred", "adversarial_pred", "prediction_adv",
        "pred_adv", "hyp_adv", "adversarial_prediction",
        "attack_pred", "attacked_pred"
    ])

    if None in [lang_key, ref_key, clean_key, adv_key]:
        print("Available keys in first row:")
        print(sorted(sample.keys()))
        raise ValueError(
            f"Key detection failed: "
            f"lang={lang_key}, ref={ref_key}, clean={clean_key}, adv={adv_key}"
        )

    print("[Detected keys]")
    print("language:", lang_key)
    print("reference:", ref_key)
    print("clean prediction:", clean_key)
    print("adversarial prediction:", adv_key)

    grouped = defaultdict(list)

    for row in rows:
        lang = str(row.get(lang_key))
        grouped[lang].append({
            "ref": row.get(ref_key, ""),
            "clean": row.get(clean_key, ""),
            "adv": row.get(adv_key, ""),
        })

    results = {}

    for lang, items in sorted(grouped.items()):
        clean_cers = []
        adv_cers = []
        clean_wers = []
        adv_wers = []

        for item in items:
            ref = item["ref"]
            clean = item["clean"]
            adv = item["adv"]

            clean_cers.append(cer(ref, clean))
            adv_cers.append(cer(ref, adv))
            clean_wers.append(wer(ref, clean))
            adv_wers.append(wer(ref, adv))

        clean_cer = safe_mean(clean_cers)
        adv_cer = safe_mean(adv_cers)
        clean_wer = safe_mean(clean_wers)
        adv_wer = safe_mean(adv_wers)

        cer_delta, cer_rel, cer_headroom = degradation(clean_cer, adv_cer)
        wer_delta, wer_rel, wer_headroom = degradation(clean_wer, adv_wer)

        results[lang] = {
            "num_samples": len(items),

            "clean_CER": clean_cer,
            "adv_CER": adv_cer,
            "CER_delta": cer_delta,
            "CER_relative_percent": cer_rel,
            "CER_headroom_percent": cer_headroom,

            "clean_WER": clean_wer,
            "adv_WER": adv_wer,
            "WER_delta": wer_delta,
            "WER_relative_percent": wer_rel,
            "WER_headroom_percent": wer_headroom,
        }

    return results


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def print_table(results):
    print("\n=== FAIR EVALUATION RESULTS ===")
    print(
        "lang\tN\tclean_CER\tadv_CER\tCER_delta\tCER_rel_%\tCER_headroom_%\t"
        "clean_WER\tadv_WER\tWER_delta\tWER_rel_%\tWER_headroom_%"
    )

    for lang, r in results.items():
        print(
            f"{lang}\t"
            f"{r['num_samples']}\t"
            f"{fmt(r['clean_CER'])}\t"
            f"{fmt(r['adv_CER'])}\t"
            f"{fmt(r['CER_delta'])}\t"
            f"{fmt(r['CER_relative_percent'])}\t"
            f"{fmt(r['CER_headroom_percent'])}\t"
            f"{fmt(r['clean_WER'])}\t"
            f"{fmt(r['adv_WER'])}\t"
            f"{fmt(r['WER_delta'])}\t"
            f"{fmt(r['WER_relative_percent'])}\t"
            f"{fmt(r['WER_headroom_percent'])}"
        )


def save_csv(results, output_path):
    import csv

    fields = [
        "language",
        "num_samples",
        "clean_CER",
        "adv_CER",
        "CER_delta",
        "CER_relative_percent",
        "CER_headroom_percent",
        "clean_WER",
        "adv_WER",
        "WER_delta",
        "WER_relative_percent",
        "WER_headroom_percent",
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
        default="./data/attack_results/fair_eval_summary.csv",
    )

    args = parser.parse_args()

    results = evaluate_jsonl(args.jsonl)
    print_table(results)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

