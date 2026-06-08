import os
import json
import argparse
import random
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

TEXT_KEYS = ["ground_truth", "raw_transcription", "transcription", "text", "sentence"]
ADV_KEYS = ["adv_wav_path", "adversarial_wav_path", "pgd_wav_path", "adv_path"]
CLEAN_KEYS = ["clean_wav_path", "original_wav_path", "orig_wav_path", "clean_path"]


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


def pick_key(row, candidates, required=True):
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    if required:
        raise KeyError(f"None of keys {candidates} found in row keys={list(row.keys())}")
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


def build_examples(rows, lang, root_dir, include_clean=True):
    examples = []

    for row in rows:
        if row.get("lang_tag") != lang:
            continue

        text = pick_key(row, TEXT_KEYS, required=True)
        adv_path = pick_key(row, ADV_KEYS, required=True)

        examples.append({
            "lang": lang,
            "wav_path": resolve_path(root_dir, adv_path),
            "text": text,
            "type": "adv",
        })

        clean_path = pick_key(row, CLEAN_KEYS, required=False)
        if include_clean and clean_path is not None:
            examples.append({
                "lang": lang,
                "wav_path": resolve_path(root_dir, clean_path),
                "text": text,
                "type": "clean",
            })

    return examples


def split_examples_by_pair(rows, lang, root_dir, include_clean=True, val_ratio=0.1, seed=42):
    """
    row 단위로 train/val split.
    clean/adv pair가 서로 다른 split으로 찢어지지 않게 함.
    """
    lang_rows = [r for r in rows if r.get("lang_tag") == lang]

    rng = random.Random(seed)
    rng.shuffle(lang_rows)

    n_total = len(lang_rows)
    n_val = max(1, int(n_total * val_ratio)) if n_total > 1 else 0

    val_rows = lang_rows[:n_val]
    train_rows = lang_rows[n_val:]

    train_examples = build_examples(train_rows, lang, root_dir, include_clean=include_clean)
    val_examples = build_examples(val_rows, lang, root_dir, include_clean=include_clean)

    return train_examples, val_examples


class WavTextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

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


def unfreeze_adapter_only(model, train_lm_head=False):
    """
    adapter_layer만 학습.
    선택적으로 lm_head도 학습.
    """
    num_trainable = 0
    trainable_names = []

    for name, p in model.named_parameters():
        name_lower = name.lower()

        train_this = False

        if "adapter_layer" in name_lower:
            train_this = True

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


def save_trainable_state(model, output_path, lang, args, step, best_val_loss=None):
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
            "best_val_loss": best_val_loss,
            "trainable_state_dict": trainable_state,
            "args": vars(args),
        },
        output_path,
    )

    print(f"[Saved] {output_path}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []

    for batch in tqdm(loader, desc="validation", leave=False):
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

        losses.append(float(outputs.loss.detach().cpu()))

    model.train()

    if len(losses) == 0:
        return None

    return sum(losses) / len(losses)


def train_one_language(lang, rows, args):
    if lang not in MMS_LANG_CODES:
        raise ValueError(f"Unsupported lang: {lang}")

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
    num_trainable, _ = unfreeze_adapter_only(model, train_lm_head=args.train_lm_head)

    if num_trainable == 0:
        raise RuntimeError("No trainable parameters found. Check adapter parameter names.")

    print_trainable_summary(model)

    train_examples, val_examples = split_examples_by_pair(
        rows=rows,
        lang=lang,
        root_dir=args.root_dir,
        include_clean=args.include_clean,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[Dataset] lang={lang}")
    print(f"  train examples: {len(train_examples)}")
    print(f"  val examples  : {len(val_examples)}")

    if len(train_examples) == 0:
        print(f"[Skip] No train examples for {lang}")
        return

    train_loader = DataLoader(
        WavTextDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    val_loader = None
    if len(val_examples) > 0:
        val_loader = DataLoader(
            WavTextDataset(val_examples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, processor),
        )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"train_log_{lang}.jsonl")

    model.train()
    global_step = 0
    best_val_loss = None

    with open(log_path, "w", encoding="utf-8") as log_f:
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader, desc=f"{lang} epoch {epoch + 1}/{args.epochs}")

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

                log_row = {
                    "lang": lang,
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": float(loss.detach().cpu()),
                    "batch_types": batch["types"],
                }

                if global_step % args.eval_every_steps == 0 and val_loader is not None:
                    val_loss = evaluate(model, val_loader, device)
                    log_row["val_loss"] = val_loss

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(args.output_dir, f"mms_adapter_ft_{lang}_best.pt")
                        save_trainable_state(
                            model=model,
                            output_path=best_path,
                            lang=lang,
                            args=args,
                            step=global_step,
                            best_val_loss=best_val_loss,
                        )

                log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                log_f.flush()

                if global_step % args.save_every_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"mms_adapter_ft_{lang}_step{global_step}.pt")
                    save_trainable_state(
                        model=model,
                        output_path=ckpt_path,
                        lang=lang,
                        args=args,
                        step=global_step,
                        best_val_loss=best_val_loss,
                    )

    final_val_loss = evaluate(model, val_loader, device) if val_loader is not None else None

    final_path = os.path.join(args.output_dir, f"mms_adapter_ft_{lang}_final.pt")
    save_trainable_state(
        model=model,
        output_path=final_path,
        lang=lang,
        args=args,
        step=global_step,
        best_val_loss=best_val_loss,
    )

    print(f"[Done] lang={lang}")
    print(f"  final checkpoint: {final_path}")
    print(f"  best val loss   : {best_val_loss}")
    print(f"  final val loss  : {final_val_loss}")
    print(f"  log             : {log_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="facebook/mms-1b-all")

    parser.add_argument("--langs", type=str, default="en,ko,ru,zh")
    parser.add_argument("--include_clean", action="store_true")
    parser.add_argument("--train_lm_head", action="store_true")

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mms_adapter_adv_ft")

    args = parser.parse_args()

    rows = read_jsonl(args.manifest)

    print(f"[Manifest] {args.manifest}")
    print(f"[Manifest] rows={len(rows)}")

    counts = defaultdict(int)
    for row in rows:
        counts[row.get("lang_tag")] += 1

    print("[Manifest language counts]")
    for lang in sorted(counts):
        print(f"  {lang}: {counts[lang]}")

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    for lang in langs:
        train_one_language(lang, rows, args)


if __name__ == "__main__":
    main()
