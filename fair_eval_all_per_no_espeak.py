import json
import csv
import argparse
import re
from collections import defaultdict

import jiwer
import jieba
import eng_to_ipa as eng_ipa
from pypinyin import lazy_pinyin, Style


# -------------------------
# Common
# -------------------------
def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def tokenize_pron_string(s):
    return " ".join(tok for tok in s.split() if tok.strip())


def safe_mean(values):
    values = [x for x in values if x is not None]
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


# -------------------------
# EN: eng_to_ipa
# -------------------------
def en_to_pron_tokens(text):
    text = normalize_text(text)
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    ipa = eng_ipa.convert(text)

    # eng_to_ipa가 모르는 단어에 *를 붙이는 경우 제거
    ipa = ipa.replace("*", "")

    # IPA를 문자 단위 token으로 사용
    # 공백은 단어 경계라 제거하고, IPA symbol 단위로 비교
    chars = [ch for ch in ipa if not ch.isspace()]
    return " ".join(chars)


# -------------------------
# KO: Hangul syllable decomposition
# -------------------------
CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


def decompose_hangul_char(ch):
    code = ord(ch)

    if not (0xAC00 <= code <= 0xD7A3):
        if ch.strip() and re.match(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", ch):
            return [ch]
        return []

    s_index = code - 0xAC00
    cho = s_index // 588
    jung = (s_index % 588) // 28
    jong = s_index % 28

    out = [CHO[cho], JUNG[jung]]
    if JONG[jong]:
        out.append(JONG[jong])

    return out


def ko_to_pron_tokens(text):
    text = normalize_text(text)
    tokens = []

    for ch in text:
        tokens.extend(decompose_hangul_char(ch))

    return " ".join(tokens)


# -------------------------
# RU: Cyrillic pronunciation-like transliteration
# -------------------------
RU_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "je",
    "ё": "jo", "ж": "ʐ", "з": "z", "и": "i", "й": "j", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts",
    "ч": "tɕ", "ш": "ʂ", "щ": "ɕː", "ъ": "", "ы": "ɨ", "ь": "",
    "э": "e", "ю": "ju", "я": "ja",
}


def ru_to_pron_tokens(text):
    text = normalize_text(text)
    tokens = []

    for ch in text:
        if ch in RU_MAP:
            val = RU_MAP[ch]
            if val:
                tokens.append(val)

    return " ".join(tokens)


# -------------------------
# ZH: Jieba + pypinyin
# -------------------------
def zh_to_pron_tokens(text):
    text = normalize_text(text)

    words = list(jieba.cut(text))
    tokens = []

    for word in words:
        pys = lazy_pinyin(
            word,
            style=Style.NORMAL,
            errors="ignore",
            strict=False,
        )
        tokens.extend(pys)

    return " ".join(tokens)


def text_to_pron_tokens(text, lang):
    if lang == "en":
        return en_to_pron_tokens(text)
    if lang == "ko":
        return ko_to_pron_tokens(text)
    if lang == "ru":
        return ru_to_pron_tokens(text)
    if lang == "zh":
        return zh_to_pron_tokens(text)

    raise ValueError(f"Unsupported lang: {lang}")


def per_from_pron_strings(ref_pron, hyp_pron):
    ref_pron = tokenize_pron_string(ref_pron)
    hyp_pron = tokenize_pron_string(hyp_pron)

    if len(ref_pron.split()) == 0:
        return None

    return jiwer.process_words(ref_pron, hyp_pron).wer


def load_grouped_rows(path, max_per_lang=None):
    grouped = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            grouped[row["lang_tag"]].append(row)

    # 중요:
    # 기존 10개 테스트 + 새 1000개가 같이 있으면 언어별 1010개가 됨.
    # 따라서 마지막 1000개만 사용.
    if max_per_lang is not None:
        grouped = {
            lang: rows[-max_per_lang:]
            for lang, rows in grouped.items()
        }

    return grouped


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
        default="./data/attack_results/all_language_per_no_espeak_summary.csv",
    )

    parser.add_argument(
        "--max_per_lang",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    grouped = load_grouped_rows(
        args.jsonl,
        max_per_lang=args.max_per_lang,
    )

    results = []

    print("=== ALL LANGUAGE PER EVALUATION WITHOUT ESPEAK ===")
    print("EN: eng_to_ipa")
    print("KO: Hangul jamo decomposition")
    print("RU: Cyrillic pronunciation-like mapping")
    print("ZH: Jieba + pypinyin")
    print()
    print("lang\tN\tclean_PER\tadv_PER\tPER_delta\tPER_rel_%\tPER_headroom_%")

    for lang in sorted(grouped):
        items = grouped[lang]

        print(f"[{lang}] converting to pronunciation tokens... N={len(items)}")

        clean_pers = []
        adv_pers = []

        for row in items:
            ref = row["ground_truth"]
            clean = row["clean_pred"]
            adv = row["adv_pred"]

            ref_pron = text_to_pron_tokens(ref, lang)
            clean_pron = text_to_pron_tokens(clean, lang)
            adv_pron = text_to_pron_tokens(adv, lang)

            clean_pers.append(per_from_pron_strings(ref_pron, clean_pron))
            adv_pers.append(per_from_pron_strings(ref_pron, adv_pron))

        clean_per = safe_mean(clean_pers)
        adv_per = safe_mean(adv_pers)
        per_delta, per_rel, per_headroom = degradation(clean_per, adv_per)

        result = {
            "language": lang,
            "num_samples": len(items),
            "clean_PER": clean_per,
            "adv_PER": adv_per,
            "PER_delta": per_delta,
            "PER_relative_percent": per_rel,
            "PER_headroom_percent": per_headroom,
        }

        results.append(result)

        print(
            f"{lang}\t"
            f"{len(items)}\t"
            f"{fmt(clean_per)}\t"
            f"{fmt(adv_per)}\t"
            f"{fmt(per_delta)}\t"
            f"{fmt(per_rel)}\t"
            f"{fmt(per_headroom)}"
        )

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "num_samples",
                "clean_PER",
                "adv_PER",
                "PER_delta",
                "PER_relative_percent",
                "PER_headroom_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Saved CSV:", args.output_csv)


if __name__ == "__main__":
    main()
