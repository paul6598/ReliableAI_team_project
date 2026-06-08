import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC


MMS_LANG_CODES = {
    "en": "eng",
    "ko": "kor",
    "ru": "rus",
    "zh": "cmn-script_simplified",
}

TEXT_KEYS = ["ground_truth", "raw_transcription", "transcription", "text", "sentence"]
CLEAN_KEYS = ["clean_wav_path", "original_wav_path", "orig_wav_path", "clean_path"]
ADV_KEYS = ["adv_wav_path", "adversarial_wav_path", "pgd_wav_path", "adv_path"]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_key(row, keys, required=True):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    if required:
        raise KeyError(f"Missing any of {keys} in row keys={list(row.keys())}")
    return None


def resolve_path(root_dir, path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(root_dir) / p)


def load_audio_16k(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    return wav.astype("float32")


def split_rows(rows, lang, val_ratio=0.1, seed=42, split="val"):
    lang_rows = [r for r in rows if r.get("lang_tag") == lang]

    rng = random.Random(seed)
    rng.shuffle(lang_rows)

    n_total = len(lang_rows)
    n_val = max(1, int(n_total * val_ratio)) if n_total > 1 else 0

    val_rows = lang_rows[:n_val]
    train_rows = lang_rows[n_val:]

    if split == "val":
        return val_rows
    if split == "train":
        return train_rows
    if split == "all":
        return lang_rows

    raise ValueError(f"Unknown split: {split}")


def set_mms_language(processor, model, lang):
    mms_code = MMS_LANG_CODES[lang]
    processor.tokenizer.set_target_lang(mms_code)

    if hasattr(model, "load_adapter"):
        model.load_adapter(mms_code)

    return mms_code


def load_trainable_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["trainable_state_dict"]

    param_dict = dict(model.named_parameters())

    loaded = 0
    missing = []

    for name, tensor in state.items():
        if name not in param_dict:
            missing.append(name)
            continue

        param_dict[name].data.copy_(tensor.to(device))
        loaded += 1

    print(f"[Checkpoint] loaded params: {loaded}")
    if missing:
        print(f"[WARNING] missing params in model: {len(missing)}")
        for name in missing[:20]:
            print("  missing:", name)

    return ckpt


@torch.no_grad()
def predict_one(model, processor, wav, device):
    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(device)

    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
    )

    pred_ids = torch.argmax(outputs.logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]

    return pred_text


def find_checkpoint(checkpoint_dir, lang, checkpoint_type):
    if checkpoint_type == "best":
        path = Path(checkpoint_dir) / f"mms_adapter_ft_{lang}_best.pt"
        if path.exists():
            return str(path)

    if checkpoint_type in ["best", "final"]:
        path = Path(checkpoint_dir) / f"mms_adapter_ft_{lang}_final.pt"
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"No checkpoint found for lang={lang}, type={checkpoint_type}, dir={checkpoint_dir}"
    )


def evaluate_lang(lang, rows, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100)
    print(f"Evaluating language: {lang}")
    print("device:", device)
    print("=" * 100)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(device)

    set_mms_language(processor, model, lang)

    if args.checkpoint_dir is not None:
        ckpt_path = find_checkpoint(args.checkpoint_dir, lang, args.checkpoint_type)
        print(f"[Checkpoint] loading: {ckpt_path}")
        load_trainable_checkpoint(model, ckpt_path, device)
    else:
        print("[Checkpoint] none. Evaluating base MMS.")

    model.eval()

    selected_rows = split_rows(
        rows=rows,
        lang=lang,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split=args.split,
    )

    if args.max_samples is not None:
        selected_rows = selected_rows[:args.max_samples]

    print(f"[Rows] {lang} {args.split}: {len(selected_rows)}")

    outputs = []

    for row in tqdm(selected_rows, desc=f"eval {lang}"):
        gt = pick_key(row, TEXT_KEYS, required=True)

        clean_path = resolve_path(args.root_dir, pick_key(row, CLEAN_KEYS, required=True))
        adv_path = resolve_path(args.root_dir, pick_key(row, ADV_KEYS, required=True))

        clean_wav = load_audio_16k(clean_path)
        adv_wav = load_audio_16k(adv_path)

        clean_pred = predict_one(model, processor, clean_wav, device)
        adv_pred = predict_one(model, processor, adv_wav, device)

        outputs.append({
            "lang_tag": lang,
            "ground_truth": gt,
            "clean_pred": clean_pred,
            "adv_pred": adv_pred,
            "clean_wav_path": clean_path,
            "adv_wav_path": adv_path,
            "split": args.split,
            "checkpoint_type": args.checkpoint_type if args.checkpoint_dir else "base",
        })

    return outputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="facebook/mms-1b-all")

    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default="best", choices=["best", "final"])

    parser.add_argument("--langs", type=str, default="en,ko,ru,zh")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--output_jsonl", type=str, required=True)

    args = parser.parse_args()

    rows = read_jsonl(args.manifest)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    all_outputs = []

    for lang in langs:
        all_outputs.extend(evaluate_lang(lang, rows, args))

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in all_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nSaved:", args.output_jsonl)
    print("Total rows:", len(all_outputs))


if __name__ == "__main__":
    main()
