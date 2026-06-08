import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC


MMS_LANG_CODES = {
    "en": "eng",
    "ko": "kor",
    "ru": "rus",
    "zh": "cmn-script_simplified",
}


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_idx}: {e}") from e

    return rows


def load_audio_16k(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    return wav.astype("float32")


class AdvWavDataset(Dataset):
    def __init__(
        self,
        rows,
        lang,
        include_clean=True,
        root_dir=".",
    ):
        self.examples = []
        self.lang = lang
        self.root_dir = Path(root_dir)

        for row in rows:
            if row["lang_tag"] != lang:
                continue

            text = row.get("ground_truth") or row.get("raw_transcription") or row.get("transcription")
            if text is None:
                raise KeyError(f"No ground truth text in row: {row}")

            adv_path = row.get("adv_wav_path") or row.get("adversarial_wav_path")
            if adv_path is None:
                raise KeyError(f"No adv_wav_path in row: {row}")

            self.examples.append({
                "wav_path": str(self.root_dir / adv_path),
                "text": text,
                "type": "adv",
            })

            clean_path = row.get("clean_wav_path")
            if include_clean and clean_path is not None:
                self.examples.append({
                    "wav_path": str(self.root_dir / clean_path),
                    "text": text,
                    "type": "clean",
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        wav = load_audio_16k(item["wav_path"])

        return {
            "wav": wav,
            "text": item["text"],
            "type": item["type"],
            "wav_path": item["wav_path"],
        }


def collate_fn(batch, processor):
    wavs = [b["wav"] for b in batch]
    texts = [b["text"] for b in batch]

    inputs = processor(
        wavs,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    labels = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.get("attention_mask", None),
        "labels": labels,
        "texts": texts,
        "types": [b["type"] for b in batch],
        "wav_paths": [b["wav_path"] for b in batch],
    }


def set_mms_language(processor, model, lang):
    mms_code = MMS_LANG_CODES[lang]

    processor.tokenizer.set_target_lang(mms_code)

    if hasattr(model, "load_adapter"):
        model.load_adapter(mms_code)

    return mms_code


def freeze_all_params(model):
    for _, p in model.named_parameters():
        p.requires_grad = False


def unfreeze_adapter_and_head(model, train_lm_head=True):
    """
    MMS-1B 전체를 학습하지 않고, language adapter와 선택적으로 lm_head만 학습한다.

    핵심:
        adapter_layer만 train
        attention q_proj/k_proj/v_proj/out_proj는 freeze
        feature_projection도 freeze
        backbone 전체 freeze
    """

    num_trainable = 0
    trainable_names = []

    for name, p in model.named_parameters():
        name_lower = name.lower()

        train_this = False

        # MMS/Wav2Vec2 adapter module만 학습
        if "adapter_layer" in name_lower:
            train_this = True

        # 선택적으로 CTC head 학습
        if train_lm_head and "lm_head" in name_lower:
            train_this = True

        p.requires_grad = train_this

        if train_this:
            num_trainable += p.numel()
            trainable_names.append(name)

    return num_trainable, trainable_names


def print_trainable_summary(model, max_names=80):
    total = 0
    trainable = 0
    names = []

    for name, p in model.named_parameters():
        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()
            names.append(name)

    print("=" * 80)
    print("[Trainable Parameters]")
    print(f"total params     : {total:,}")
    print(f"trainable params : {trainable:,}")
    print(f"trainable ratio  : {100 * trainable / total:.4f}%")
    print("-" * 80)

    for name in names[:max_names]:
        print("  ", name)

    if len(names) > max_names:
        print(f"  ... and {len(names) - max_names} more")

    print("=" * 80)


def save_trainable_state(model, output_path, lang, args, step):
    """
    전체 MMS-1B를 저장하면 너무 크므로, requires_grad=True인 파라미터만 저장한다.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    trainable_state = {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    torch.save(
        {
            "lang": lang,
            "step": step,
            "trainable_state_dict": trainable_state,
            "args": vars(args),
        },
        output_path,
    )

    print(f"[Saved trainable adapter/head checkpoint] {output_path}")


def train_one_language(lang, rows, args):
    if lang not in MMS_LANG_CODES:
        raise ValueError(f"Unsupported lang: {lang}. Available: {sorted(MMS_LANG_CODES)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100)
    print(f"Fine-tuning MMS adapter for language: {lang}")
    print("device:", device)
    print("=" * 100)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(device)

    mms_code = set_mms_language(processor, model, lang)
    print(f"[MMS] lang={lang}, mms_code={mms_code}")

    freeze_all_params(model)

    num_trainable, trainable_names = unfreeze_adapter_and_head(
        model,
        train_lm_head=args.train_lm_head,
    )

    if num_trainable == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Adapter parameter names may differ in this transformers version."
        )

    print_trainable_summary(model)

    dataset = AdvWavDataset(
        rows=rows,
        lang=lang,
        include_clean=args.include_clean,
        root_dir=args.root_dir,
    )

    print(f"[Dataset] lang={lang}, examples={len(dataset)}")

    if len(dataset) == 0:
        print(f"[Skip] No examples for lang={lang}")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model.train()

    global_step = 0
    log_path = os.path.join(args.output_dir, f"train_log_{lang}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"{lang} epoch {epoch + 1}/{args.epochs}")

            for batch in pbar:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                attention_mask = batch["attention_mask"]
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )
                optimizer.step()

                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": global_step,
                })

                row = {
                    "lang": lang,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "batch_types": batch["types"],
                }

                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_f.flush()

                if global_step % args.save_every_steps == 0:
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"mms_adapter_ft_{lang}_step{global_step}.pt",
                    )
                    save_trainable_state(model, ckpt_path, lang, args, global_step)

    final_path = os.path.join(args.output_dir, f"mms_adapter_ft_{lang}_final.pt")
    save_trainable_state(model, final_path, lang, args, global_step)

    print(f"[Done] lang={lang}, final checkpoint={final_path}")
    print(f"[Log] {log_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, default="./data/adv_wavs/manifest.jsonl")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="facebook/mms-1b-all")

    parser.add_argument("--langs", type=str, default="en,ko,ru,zh")
    parser.add_argument("--include_clean", action="store_true")
    parser.add_argument("--train_lm_head", action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mms_adapter_adv_ft")

    args = parser.parse_args()

    rows = read_jsonl(args.manifest)

    print(f"[Manifest] {args.manifest}")
    print(f"[Manifest] rows={len(rows)}")

    counts = defaultdict(int)
    for row in rows:
        counts[row["lang_tag"]] += 1

    print("[Manifest language counts]")
    for lang in sorted(counts):
        print(f"  {lang}: {counts[lang]}")

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    for lang in langs:
        train_one_language(lang, rows, args)


if __name__ == "__main__":
    main()
