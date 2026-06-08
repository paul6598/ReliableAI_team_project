import os
import json
import numpy as np
import soundfile as sf
from collections import defaultdict
from datasets import load_from_disk

from src.data_load import load_equalized_dataset


def load_dataset_for_fake(samples_per_lang):
    saved_path = "./data/waveform"

    if os.path.isdir(saved_path):
        print(f"[Dataset] Loading saved dataset from {saved_path}")
        dataset = load_from_disk(saved_path)
    else:
        print("[Dataset] Saved dataset not found. Loading FLEURS...")
        dataset = load_equalized_dataset(
            dataset_name="google/fleurs",
            split="train",
            languages=["ko", "en", "zh", "ru"],
            samples_per_lang=max(samples_per_lang, 1000),
        )

    return dataset


def get_text(sample):
    for key in ["raw_transcription", "transcription", "ground_truth", "text"]:
        if key in sample and sample[key] is not None:
            return sample[key]
    raise KeyError(f"No text key found: {list(sample.keys())}")


def main():
    samples_per_lang = 5
    langs = ["en", "ko", "ru", "zh"]

    output_root = "./data/fake_adv_wavs"
    manifest_path = os.path.join(output_root, "manifest.jsonl")

    os.makedirs(output_root, exist_ok=True)

    dataset = load_dataset_for_fake(samples_per_lang)

    groups = defaultdict(list)

    for sample in dataset:
        lang = sample["lang_tag"]
        if lang in langs:
            groups[lang].append(sample)

    rows = []

    rng = np.random.default_rng(42)

    for lang in langs:
        lang_dir = os.path.join(output_root, lang)
        os.makedirs(lang_dir, exist_ok=True)

        selected = groups[lang][:samples_per_lang]

        print(f"[{lang}] selected {len(selected)} samples")

        for i, sample in enumerate(selected):
            audio = sample["audio"]
            wav = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])

            if sr != 16000:
                raise RuntimeError(f"Expected 16kHz, got {sr}")

            text = get_text(sample)

            clean_path = os.path.join(lang_dir, f"{i:04d}_clean.wav")
            adv_path = os.path.join(lang_dir, f"{i:04d}_fake_adv.wav")

            # fake adversarial: 아주 작은 noise만 추가
            noise = rng.normal(0.0, 0.0005, size=wav.shape).astype(np.float32)
            adv = np.clip(wav + noise, -1.0, 1.0).astype(np.float32)

            sf.write(clean_path, wav, 16000)
            sf.write(adv_path, adv, 16000)

            rows.append({
                "lang_tag": lang,
                "ground_truth": text,
                "clean_wav_path": clean_path,
                "adv_wav_path": adv_path,
                "is_fake_adv": True,
            })

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"Saved manifest: {manifest_path}")
    print(f"Total rows: {len(rows)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
