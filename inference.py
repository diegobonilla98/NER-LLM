# infer_ner_adapters.py - Supports both LoRA and DoRA adapters
import os, re, math, json, tempfile, shutil
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ───────────────────────────────────────────────────────────────────────────────
# Prompt (identical shape to training)
# ───────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a precise NER tagger.\n"
    "- Output ONLY a JSON object mapping exact surface strings to their entity types.\n"
    "- Keys must be substrings found in the input text, with identical characters and spacing.\n"
    "- Do not add explanations or any extra text. Return valid JSON only."
)
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    f"{SYSTEM_INSTRUCTION}\n\n"
    "### Input:\n{input}\n\n"
    "### Output:\n"
)

# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────
BOE_RE = re.compile(r"<boe:\s*([A-Za-z0-9_\-]+)>", flags=re.IGNORECASE)
EOE = "<eoe>"

def extract_entities_from_tagged(tagged: str) -> Tuple[str, List[Dict]]:
    """
    Parse `<boe: TYPE> ... <eoe>` tags from a tagged string.
    Returns (plain_text_without_tags, entities) where each entity:
      {"start": int, "end": int, "type": str, "text": str}
    Offsets (start, end) are in the plain_text coordinates.
    """
    entities = []
    plain_parts = []
    i = 0
    plain_pos = 0
    n = len(tagged)

    while i < n:
        m = BOE_RE.search(tagged, i)
        if not m:
            # no more tags; copy the rest
            chunk = tagged[i:]
            plain_parts.append(chunk)
            plain_pos += len(chunk)
            break

        # copy non-entity chunk
        pre = tagged[i:m.start()]
        plain_parts.append(pre)
        plain_pos += len(pre)

        # read begin tag and type
        ent_type = m.group(1).strip()
        j = m.end()

        # find end tag
        e_idx = tagged.find(EOE, j)
        if e_idx == -1:
            # malformed (no closing tag) — treat the rest as plain
            rest = tagged[m.start():]
            plain_parts.append(rest)
            plain_pos += len(rest)
            break

        ent_text = tagged[j:e_idx]
        start = plain_pos
        end = start + len(ent_text)
        entities.append({"start": start, "end": end, "type": ent_type, "text": ent_text})

        # add the entity text itself (tags removed) to the plain buffer
        plain_parts.append(ent_text)
        plain_pos += len(ent_text)

        # advance after the <eoe>
        i = e_idx + len(EOE)

    plain_text = "".join(plain_parts)
    return plain_text, entities


def _sanitize_adapter_directory(adapter_dir: str) -> str:
    """
    Remove unsupported keys (e.g., 'corda_config') from adapter_config.json by copying
    the adapter to a temporary directory with a sanitized config. Returns the directory
    to load adapters from (original or sanitized temp dir).
    """
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return adapter_dir
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return adapter_dir

    removed = []
    # Peft LoraConfig fields allowlist (commonly used across versions)
    allowlist = {
        "peft_type",
        "base_model_name_or_path",
        "revision",
        "task_type",
        "inference_mode",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "modules_to_save",
        "init_lora_weights",
        "bias",
        "fan_in_fan_out",
        "use_rslora",
        "use_dora",
        "rank_pattern",
        "alpha_pattern",
        "layers_to_transform",
        "layers_pattern",
        "megatron_config",
        "tensor_parallel_size",
        "loftq_config",
        "task_specific_params",
    }

    # Drop any keys not in allowlist
    for k in list(cfg.keys()):
        if k not in allowlist:
            cfg.pop(k, None)
            removed.append(k)

    if not removed:
        return adapter_dir

    tmp_dir = tempfile.mkdtemp(prefix="peft_adapter_sanitized_")
    # Copy adapter weights
    for fname in ["adapter_model.safetensors", "adapter_model.bin", "README.md", "README.txt"]:
        src = os.path.join(adapter_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tmp_dir, fname))
    # Write sanitized config
    with open(os.path.join(tmp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Sanitized adapter config: removed unsupported keys {removed}; using temp dir: {tmp_dir}")
    return tmp_dir


def _extract_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object substring from text using brace matching.
    """
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _parse_json_mapping(generated: str) -> Dict[str, str]:
    """
    Parse a JSON object mapping surface strings to entity types. Returns empty dict on failure.
    """
    snippet = _extract_json_object(generated.strip())
    if not snippet:
        try:
            return {str(k): str(v) for k, v in json.loads(generated).items()}
        except Exception:
            return {}
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        return {}
    except Exception:
        return {}


def _find_all_occurrences(haystack: str, needle: str) -> List[int]:
    """
    Return all start indices where needle occurs in haystack (including overlapping).
    """
    if not needle:
        return []
    starts: List[int] = []
    i = 0
    while True:
        j = haystack.find(needle, i)
        if j == -1:
            break
        starts.append(j)
        i = j + 1
    return starts


def _build_entities_from_mapping(text: str, mapping: Dict[str, str]) -> Tuple[List[Dict], List[str]]:
    """
    Convert a string->type mapping into non-overlapping entity spans on the given text.
    Preference is given to longer matches to avoid fragment overlaps.
    Returns (entities, unmatched_strings).
    """
    candidates: List[Tuple[int, int, str, str]] = []
    unmatched: List[str] = []

    for surface, ent_type in mapping.items():
        positions = _find_all_occurrences(text, surface)
        if not positions:
            unmatched.append(surface)
            continue
        for s in positions:
            e = s + len(surface)
            candidates.append((s, e, ent_type, surface))

    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

    occupied = [False] * len(text)
    entities: List[Dict] = []
    for s, e, ent_type, surface in candidates:
        if s < 0 or e > len(text) or s >= e:
            continue
        if any(occupied[s:e]):
            continue
        for idx in range(s, e):
            occupied[idx] = True
        entities.append({"start": s, "end": e, "type": ent_type, "text": surface})

    entities.sort(key=lambda d: (d["start"], d["end"]))
    return entities, unmatched


def load_model_and_tokenizer(model_name: str, adapter_dir: str, dtype: str = "auto"):
    """
    Load base model and attach LoRA or DoRA adapters.
    Automatically detects adapter type from configuration.
    Prefer bfloat16 on modern GPUs; fallback to float16; CPU uses float32.
    """
    if dtype == "auto":
        if torch.cuda.is_available():
            # bf16 if supported; otherwise fp16
            torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter configuration to detect type
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        adapter_type = adapter_config.get("peft_type", "LORA")
        use_dora = adapter_config.get("use_dora", False)
        print(f"Detected adapter type: {adapter_type}" + (f" (DoRA enabled)" if use_dora else ""))
    else:
        print("Warning: adapter_config.json not found, assuming LoRA")
        adapter_type = "LORA"
        use_dora = False

    # Proactively clear CUDA cache between loads to avoid stale state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load PEFT adapter (works for both LoRA and DoRA)
    sanitized_dir = _sanitize_adapter_directory(adapter_dir)
    model = PeftModel.from_pretrained(base, sanitized_dir, device_map="auto")
    model.eval()
    # Fast decode
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return model, tokenizer


@torch.inference_mode()
def tag_text(model, tokenizer, text: str, max_seq_len: int = 2048, advanced: bool = False) -> Dict:
    """
    Build the prompt, generate deterministically, and parse a JSON mapping
    from surface string -> entity type. Then map to character spans.
    """
    prompt = PROMPT_TEMPLATE.format(input=text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    # place on the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Conservative, context-safe max_new_tokens:
    # allow up to the remaining room for tags, with a floor and ceiling.
    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = max(64, min(1024, max_seq_len - prompt_len - 4))

    if advanced:
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_beams=1,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    else:
        gen_ids = model.generate(
            **inputs,
            do_sample=False,            # deterministic decode for strict JSON formatting
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the newly generated suffix
    new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Parse JSON mapping and project to spans
    mapping = _parse_json_mapping(raw_output)
    entities, unmatched = _build_entities_from_mapping(text, mapping)

    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "mapping": mapping,
        "entities": entities,
        "unmatched_strings": unmatched,
    }


def main():
    # User-defined configuration variables
    model_name = "Qwen/Qwen3-0.6B"
    adapter_dir = r"ner_checkpoint_dorav2\qwen3-0p6b-ner-lora\best_lora"  # Works with both LoRA and DoRA adapters
    max_seq_len = 2048

    model, tokenizer = load_model_and_tokenizer(model_name, adapter_dir)

    text = (
        "Factura 2024-0098 emitida a Apple Retail Spain, S.L.U., "
        "Calle Príncipe de Vergara 112, 28002 Madrid. "
        "Contacto: john.doe@example.com, +34 600 123 456."
    )

    out = tag_text(model, tokenizer, text, max_seq_len=max_seq_len)

    print("\n=== Model Raw Output ===")
    print(out["raw_output"])
    print("\n=== Parsed JSON Mapping ===")
    print(json.dumps(out.get("mapping", {}), ensure_ascii=False, indent=2))
    print("\n=== Entities (start, end, type, text) ===")
    for ent in out["entities"]:
        print(f"{ent['start']:>4}-{ent['end']:<4}  {ent['type']:<12}  {repr(ent['text'])}")

    if out.get("unmatched_strings"):
        print("\n[WARN] Some keys in the JSON mapping were not found in the input text:")
        for s in out["unmatched_strings"]:
            print(f"  - {repr(s)}")

if __name__ == "__main__":
    main()
