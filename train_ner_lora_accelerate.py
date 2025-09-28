import os
# --- Silence HF tokenizers fork warnings BEFORE any HF import ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json, math, time, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from tqdm.auto import tqdm


# =========================
# Prompt (pure text format)
# =========================
SYSTEM_INSTRUCTION = (
    "You are a precise NER tagger. Keep characters and spacing identical to the input. "
    "Only insert inline tags <boe:TYPE> and <eoe> around exact entity spans. "
    "Do not paraphrase or remove any characters."
)
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    f"{SYSTEM_INSTRUCTION}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_json_loads(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        x = x.strip()
        try: return json.loads(x)
        except Exception: return None
    return None

def non_overlapping_insertions(text: str, matches: List[Tuple[int,int,str,str]]) -> str:
    out, cur = [], 0
    for s, e, typ, span in matches:
        if s < cur:  # skip overlaps
            continue
        out.append(text[cur:s])
        out.append(f"<boe: {typ}>{text[s:e]}<eoe>")
        cur = e
    out.append(text[cur:])
    return "".join(out)

def build_tagged_output_from_map(text: str, span2type: Dict[str, str]) -> str:
    if not span2type: return text
    taken, chosen = [], []
    items = sorted(span2type.items(), key=lambda kv: (-len(kv[0]), kv[0]))  # longer first
    for span, typ in items:
        if not span: continue
        start = 0
        while True:
            idx = text.find(span, start)
            if idx == -1: break
            s, e = idx, idx + len(span)
            if any(not (e <= ts or s >= te) for ts, te in taken):
                start = e; continue
            taken.append((s, e)); chosen.append((s, e, typ, span))
            break
    chosen.sort(key=lambda t: t[0])
    return non_overlapping_insertions(text, chosen)


# =========================
# Build prompt/target pair
# =========================
def build_texts(example: Dict[str, str]) -> Dict[str, str]:
    raw_inp = example["input"]
    prompt = PROMPT_TEMPLATE.format(input=raw_inp)

    ent_map = None
    if "output" in example:
        ent_map = safe_json_loads(example["output"])
    if ent_map is None and "entities" in example:
        ent_map = safe_json_loads(example["entities"])

    if ent_map is not None:
        target_text = build_tagged_output_from_map(raw_inp, ent_map)
    else:
        target_text = example.get("output", raw_inp)  # already-tagged fallback

    return {"prompt": prompt, "target": target_text}


# =========================
# Tokenize (single trunc)
# =========================
def tokenize_one(example, tokenizer, max_len: int):
    # 1) tokenize prompt (no EOS) + target (+EOS) WITHOUT truncation
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(example["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # 2) concat then single LEFT truncation
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_len:
        cut = len(input_ids) - max_len
        input_ids = input_ids[cut:]
        labels = labels[cut:]
        attention_mask = attention_mask[cut:]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# =========================
# Packing (concatenate many)
# =========================
def pack_features(ds: Dataset, max_len: int, pad_id: int, pad_label: int = -100) -> Dataset:
    buf_ids, buf_labels, buf_mask = [], [], []
    out_ids, out_labels, out_mask = [], [], []

    for ex in ds:
        ids, labels, mask = ex["input_ids"], ex["labels"], ex["attention_mask"]
        i = 0
        while i < len(ids):
            space = max_len - len(buf_ids)
            take = min(space, len(ids) - i)
            buf_ids.extend(ids[i:i+take])
            buf_labels.extend(labels[i:i+take])
            buf_mask.extend(mask[i:i+take])
            i += take
            if len(buf_ids) == max_len:
                out_ids.append(buf_ids); out_labels.append(buf_labels); out_mask.append(buf_mask)
                buf_ids, buf_labels, buf_mask = [], [], []
    if buf_ids:
        pad = max_len - len(buf_ids)
        out_ids.append(buf_ids + [pad_id] * pad)
        out_labels.append(buf_labels + [pad_label] * pad)
        out_mask.append(buf_mask + [0] * pad)

    return Dataset.from_dict({"input_ids": out_ids, "labels": out_labels, "attention_mask": out_mask})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/qwen3-0p6b-ner-lora")
    p.add_argument("--max_seq_len", type=int, default=2048)  # packing window
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_bs", type=int, default=2)   # micro-batch per device
    p.add_argument("--eval_bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Precision / DS
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--ds_config", type=str, default="ds_zero2.json")
    args = p.parse_args()
    set_seed(args.seed)

    # --- DeepSpeed plugin (file-based, no interactive prompts) ---
    ds_plugin = DeepSpeedPlugin(hf_ds_config=args.ds_config) if os.path.isfile(args.ds_config) else None
    mixed = "bf16" if args.bf16 else ("fp16" if args.fp16 else "no")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mixed,
        deepspeed_plugin=ds_plugin,
        log_with=None,
    )
    is_main = accelerator.is_main_process
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"World size: {accelerator.state.num_processes} | Mixed precision: {mixed} | DS: {bool(ds_plugin)}")

    # --- Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, trust_remote_code=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM, use_dora=True
    )
    model = get_peft_model(model, lora_cfg)

    # --- Data: JSONL -> prompt/target -> tokenize -> pack ---
    raw_train = load_dataset("json", data_files=args.train_file, split="train")
    raw_val   = load_dataset("json", data_files=args.val_file,   split="train")

    train_ft = raw_train.map(build_texts, remove_columns=raw_train.column_names, desc="Build prompts/targets [train]")
    val_ft   = raw_val.map(build_texts,   remove_columns=raw_val.column_names,   desc="Build prompts/targets [val]")

    def _tok(ex): return tokenize_one(ex, tokenizer, args.max_seq_len)
    train_tok = train_ft.map(_tok, remove_columns=train_ft.column_names, desc="Tokenize [train]")
    val_tok   = val_ft.map(_tok,   remove_columns=val_ft.column_names,   desc="Tokenize [val]")

    train_pack = pack_features(train_tok, args.max_seq_len, tokenizer.pad_token_id, -100)
    val_pack   = pack_features(val_tok,   args.max_seq_len, tokenizer.pad_token_id, -100)

    # Return tensors already; let default collate stack them (no custom collate_fn)
    train_pack.set_format(type="torch", columns=["input_ids","labels","attention_mask"])
    val_pack.set_format(type="torch", columns=["input_ids","labels","attention_mask"])

    # Databricks / fork-safe: num_workers=0 (no workers), avoid the TypeError path
    train_loader = DataLoader(train_pack, batch_size=args.train_bs, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_pack,   batch_size=args.eval_bs,  shuffle=False, drop_last=False, num_workers=0)

    # --- Optimizer & Scheduler ---
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9,0.95), weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)  # real optimizer steps per epoch
    max_train_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max_train_steps)

    # --- Prepare distributed ---
    model, optim, train_loader, val_loader, sched = accelerator.prepare(model, optim, train_loader, val_loader, sched)

    # --- Train ---
    global_step, best_val = 0, float("inf")
    if is_main:
        print(f"Steps/epoch (optimizer steps): {steps_per_epoch} | Max steps: {max_train_steps} | Warmup: {warmup_steps}")

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        bar = tqdm(total=len(train_loader), disable=not is_main, desc=f"Epoch {epoch+1}/{args.epochs} [train]")

        loss_sum_for_log = 0.0
        tokens_sum_for_log = 0
        time_sum = 0.0
        last_t = time.time()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # global loss for logging
                with torch.no_grad():
                    gathered = accelerator.gather(loss.detach())
                    loss_sum_for_log += gathered.mean().float().item()

                # tokens (global) contributing to loss
                mb_tokens = (batch["labels"] != -100).sum()
                mb_tokens = accelerator.gather(mb_tokens).sum().item()
                tokens_sum_for_log += int(mb_tokens)

                if accelerator.sync_gradients:
                    optim.step()
                    optim.zero_grad()
                    sched.step()  # step LR only on real optimizer steps
                    global_step += 1

                # timings / progress
                now = time.time()
                time_sum += (now - last_t)
                last_t = now
                denom = (step + 1)
                toks_per_sec = (tokens_sum_for_log / time_sum) if time_sum > 0 else 0.0
                bar.set_postfix({"loss": f"{(loss_sum_for_log/denom):.4f}", "t/s": f"{toks_per_sec:,.0f}"})
            bar.update(1)

        bar.close()
        epoch_time = time.time() - epoch_start

        # --- Eval ---
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            vbar = tqdm(total=len(val_loader), disable=not is_main, desc=f"Epoch {epoch+1}/{args.epochs} [eval]")
            for batch in val_loader:
                outputs = model(**batch)
                loss = outputs.loss
                gathered = accelerator.gather(loss.detach())
                val_loss_sum += gathered.mean().float().item()
                val_steps += 1
                vbar.update(1)
            vbar.close()

        avg_train_loss = loss_sum_for_log / max(1, len(train_loader))
        avg_val_loss = val_loss_sum / max(1, val_steps)
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")

        if is_main:
            print(f"[Epoch {epoch+1}] time={epoch_time:.1f}s  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  ppl={ppl:.2f}")

        # save best
        if is_main and avg_val_loss < best_val:
            best_val = avg_val_loss
            save_dir = os.path.join(args.output_dir, "best_lora")
            accelerator.unwrap_model(model).save_pretrained(save_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best LoRA adapters to: {save_dir}")

    # final save
    if is_main:
        final_dir = os.path.join(args.output_dir, "last_lora")
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved last LoRA adapters to: {final_dir}")


if __name__ == "__main__":
    main()
