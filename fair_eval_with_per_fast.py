import json
import argparse
from pathlib import Path
from collections import defaultdict

import jiwer
from phonemizer import phonemize
from phonemizer.separator import Separator


LANG2ESPEAK = {
    "en": "en-us",
    "ko": "ko",
    "ru": "ru",
    "zh": "cmn",
}

WORD_MARK = "▁"

PHONE_SEP = Separator(
    phone=" ",
    word=f" {WORD_MARK} ",
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def strip_word_marks(phone_string: str) -> str:
    return " ".join(tok for tok in phone_string.split() if tok != WORD_MARK)


def batch_phonemize(texts, lang, njobs=4):
    texts = [normalize_text(x) for x in texts]

    if lang not in LANG2ESPEAK:
        raise ValueError(f"Unknown language for phonemizer: {lang}")

    phones = phonemize(
        texts,
        language=LANG2ESPEAK[lang],
        backend="espeak",
        separator=PHONE_SEP,
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
        njobs=njobs,
    )

    return [strip_word_marks(x) for x in phones]


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


def per_from_phone(ref_phone: str, hyp_phone: str):
    if len(ref_phone.split()) == 0:
        return None

    return jiwer.process_words(ref_phone, hyp_phone).wer


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


def evaluate_jsonl(path, njobs=4):
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

        refs = [x["ref"] for x in items]
        cleans = [x["clean"] for x in items]
        advs = [x["adv"] for x in items]

        print(f"  {lang}: computing CER/WER")
        clean_cers = [cer(r, h) for r, h in zip(refs, cleans)]
        adv_cers = [cer(r, h) for r, h in zip(refs, advs)]
        clean_wers = [wer(r, h) for r, h in zip(refs, cleans)]
        adv_wers = [wer(r, h) for r, h in zip(refs, advs)]

        if lang == "zh":
            print(f"  {lang}: skipping PER because espeak-ng Chinese dictionary is unstable/missing")
            clean_pers = [None] * len(items)
            adv_pers = [None] * len(items)
        else:
            print(f"  {lang}: phonemizing refs")
            ref_phones = batch_phonemize(refs, lang, njobs=njobs)

            print(f"  {lang}: phonemizing clean predictions")
            clean_phones = batch_phonemize(cleans, lang, njobs=njobs)

            print(f"  {lang}: phonemizing adversarial predictions")
            adv_phones = batch_phonemize(advs, lang, njobs=njobs)

            print(f"  {lang}: computing PER")
            clean_pers = [
                per_from_phone(rp, hp)
                for rp, hp in zip(ref_phones, clean_phones)
            ]
            adv_pers = [
                per_from_phone(rp, hp)
                for rp, hp in zip(ref_phones, adv_phones)
            ]

        clean_cer = safe_mean(clean_cers)
        adv_cer = safe_mean(adv_cers)
        clean_wer = safe_mean(clean_wers)
        adv_wer = safe_mean(adv_wers)
        clean_per = safe_mean(clean_pers)
        adv_per = safe_mean(adv_pers)

        cer_delta, cer_rel, cer_headroom = degradation(clean_cer, adv_cer)
        wer_delta, wer_rel, wer_headroom = degradation(clean_wer, adv_wer)
        per_delta, per_rel, per_headroom = degradation(clean_per, adv_per)

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

            "clean_PER": clean_per,
            "adv_PER": adv_per,
            "PER_delta": per_delta,
            "PER_relative_percent": per_rel,
            "PER_headroom_percent": per_headroom,
        }

    return results


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def print_table(results):
    print("\n=== FAIR EVALUATION RESULTS WITH PER ===")
    print(
        "lang\tN\t"
        "clean_CER\tadv_CER\tCER_delta\tCER_rel_%\tCER_headroom_%\t"
        "clean_PER\tadv_PER\tPER_delta\tPER_rel_%\tPER_headroom_%"
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
            f"{fmt(r['clean_PER'])}\t"
            f"{fmt(r['adv_PER'])}\t"
            f"{fmt(r['PER_delta'])}\t"
            f"{fmt(r['PER_relative_percent'])}\t"
            f"{fmt(r['PER_headroom_percent'])}"
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

        "clean_PER",
        "adv_PER",
        "PER_delta",
        "PER_relative_percent",
        "PER_headroom_percent",
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
        default="./data/attack_results/fair_eval_with_per_summary.csv",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    results = evaluate_jsonl(args.jsonl, njobs=args.njobs)
    print_table(results)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
