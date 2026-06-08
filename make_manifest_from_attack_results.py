import json
from pathlib import Path
from collections import defaultdict, Counter

input_jsonl = Path("./data/attack_results/all_results.jsonl")
output_jsonl = Path("./data/attack_results/manifest.jsonl")
wav_root = Path("./data/attack_results")

grouped = defaultdict(list)

with input_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            grouped[row["lang_tag"]].append(row)

rows_out = []
written = Counter()

for lang, rows in grouped.items():
    # 기존 10개 테스트가 앞에 있으므로 마지막 1000개만 사용
    rows = rows[-1000:]

    for idx, row in enumerate(rows):
        clean_path = wav_root / lang / "orig" / f"{lang}_{idx:04d}_clean.wav"
        adv_path = wav_root / lang / "adv" / f"{lang}_{idx:04d}_adv.wav"

        if not clean_path.exists() or not adv_path.exists():
            print("[SKIP missing]", lang, idx, clean_path, adv_path)
            continue

        rows_out.append({
            "lang_tag": lang,
            "ground_truth": row["ground_truth"],
            "clean_wav_path": str(clean_path),
            "adv_wav_path": str(adv_path),
        })
        written[lang] += 1

with output_jsonl.open("w", encoding="utf-8") as f:
    for row in rows_out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved:", output_jsonl)
print("Written:", dict(written))
print("Total:", len(rows_out))
