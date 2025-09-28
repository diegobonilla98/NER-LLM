from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gradio as gr
import gc
import torch

# Import your inference helpers
from inference import load_model_and_tokenizer, tag_text  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class UIConfig:
    title: str = "🧠✨ EntityScope — Smart NER Highlighter"
    description: str = (
        "**EntityScope** wraps your custom NER LLM. The model outputs ONLY a JSON object mapping "
        "exact surface strings to entity types (keys must be substrings of the input). "
        "We parse the mapping and render a highlighted view with labels. No hand-written rules, "
        "no traditional NER — just your model."
    )
    default_model: str = "Qwen/Qwen3-0.6B"
    default_adapter: str = r"ner_checkpoint_dorav2\qwen3-0p6b-ner-lora\best_lora"
    default_dtype: str = "auto"   # auto | float16 | bfloat16 | float32
    default_max_input_tokens: int = 512
    default_max_seq_len: int = 2048


CFG = UIConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> Tuple[str, int, bool]:
    """Return (possibly truncated) text, token_count, was_truncated."""
    if max_tokens is None or max_tokens <= 0:
        return text, 0, False
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    if n <= max_tokens:
        return text, n, False
    # Truncate by tokens, then decode back
    truncated_ids = ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, n, True


def _entities_to_spans(text: str, entities: List[Dict]) -> List[Tuple[str, str | None]]:
    """Convert entity offsets into HighlightedText spans.
    Returns a list of (segment_text, label_or_None), covering the entire text.
    """
    if not entities:
        return [(text, None)] if text else []

    entities = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])) )
    spans: List[Tuple[str, str | None]] = []
    i = 0
    for ent in entities:
        s, e, t = int(ent["start"]), int(ent["end"]), str(ent["type"])
        if s > i:
            spans.append((text[i:s], None))
        # clamp bounds defensively
        s = max(0, min(s, len(text)))
        e = max(0, min(e, len(text)))
        if e > s:
            spans.append((text[s:e], t))
        i = max(i, e)
    if i < len(text):
        spans.append((text[i:], None))
    return spans


# ──────────────────────────────────────────────────────────────────────────────
# Model state & actions
# ──────────────────────────────────────────────────────────────────────────────
class ModelBundle:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None


MB = ModelBundle()


def init_model(model_name: str, adapter_dir: str, dtype: str) -> str:
    """Load base model + adapters into global bundle."""
    # Clean previous model/tokenizer and GPU caches between runs
    if MB.ready:
        try:
            del MB.model
            del MB.tokenizer
        except Exception:
            pass
        MB.model, MB.tokenizer = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    MB.model, MB.tokenizer = load_model_and_tokenizer(model_name, adapter_dir, dtype=dtype)
    return f"Loaded: {model_name}\nAdapter: {adapter_dir}\nDType: {dtype}"


def run_ner(
    text: str,
    max_input_tokens: int,
    max_seq_len: int,
    advanced: bool,
) -> Tuple[List[Tuple[str, str | None]], str]:
    if not MB.ready:
        raise gr.Error("Model is not loaded yet. Click ‘Load / Reload model’ in Advanced settings.")

    t_text, total_tokens, was_trunc = _truncate_to_tokens(text, MB.tokenizer, max_input_tokens)

    out = tag_text(MB.model, MB.tokenizer, t_text, max_seq_len=max_seq_len, advanced=advanced)
    # Build spans for UI
    entities = out.get("entities", [])
    spans = _entities_to_spans(t_text, entities)

    # Compose status message
    msgs = []
    if was_trunc:
        msgs.append(
            f"⚠️ Input was truncated to {max_input_tokens} tokens (original: {total_tokens})."
        )
    unmatched = out.get("unmatched_strings", [])
    if unmatched:
        preview = ", ".join(repr(s) for s in unmatched[:5])
        more = "…" if len(unmatched) > 5 else ""
        msgs.append(
            f"⚠️ {len(unmatched)} mapping key(s) not found in input: {preview}{more}"
        )
    mapping = out.get("mapping", {})
    meta = (
        f"Mapping keys: {len(mapping)}. Matched entities: {len(entities)}. "
        f"Unmatched: {len(unmatched)}. Max new tokens used: ≤ {max_seq_len}."
    )
    status = ("\n".join(msgs) + ("\n" if msgs else "") + meta).strip()

    return spans, status


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css="""
:root { --radius-xl: 18px; }
.gradio-container { max-width: 980px !important; margin: auto; }
#app-title h1 { font-weight: 800; letter-spacing: -0.02em; }
footer { visibility: hidden; }
.card { border-radius: 18px; }
""") as demo:
    gr.Markdown(f"""
    <div id="app-title">
      <h1>{CFG.title}</h1>
      <p style="font-size:1.02rem; color:#475569;">{CFG.description}</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Input text",
                placeholder=(
                    "Paste text (contracts, invoices, emails, logs…). The model will output ONLY a JSON "
                    "object mapping exact substrings to entity types."
                ),
                lines=12,
                autofocus=True,
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    step=16,
                    value=CFG.default_max_input_tokens,
                    label="Max input tokens (pre‑prompt)",
                    info="Input is tokenized with the model tokenizer and truncated to this length.",
                )
                max_seq_len = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    step=64,
                    value=CFG.default_max_seq_len,
                    label="Max generation context (new tokens cap)",
                )
            advanced = gr.Checkbox(value=False, label="Advanced generation", info="Enable sampling (temperature, top‑p, top‑k) for wider search.")
            run_btn = gr.Button("🔎 Extract entities", variant="primary")

        with gr.Column(scale=3):
            highlighted = gr.HighlightedText(
                label="Recognized entities",
                combine_adjacent=True,
                show_legend=True,
            )
            status_out = gr.Markdown(label="Status & Warnings")

    with gr.Accordion("Advanced settings", open=False):
        with gr.Row():
            model_name = gr.Textbox(value=CFG.default_model, label="Base model name or path")
            adapter_dir = gr.Dropdown(
                choices=[
                    r"ner_checkpoint_dorav2\qwen3-0p6b-ner-lora\best_lora",
                    r"ner_checkpoint_dora\qwen3-0p6b-ner-lora\best_lora",
                    r"ner_checkpoint\qwen3-0p6b-ner-lora\best_lora",
                ],
                value=CFG.default_adapter,
                label="Adapter directory (LoRA/DoRA)",
                info="Choose between DoRA/LoRA checkpoints",
            )
            dtype = gr.Dropdown(
                choices=["auto", "float16", "bfloat16", "float32"],
                value=CFG.default_dtype,
                label="Precision",
                info="auto = bf16 on new CUDA GPUs, else fp16; CPU uses fp32",
            )
        load_btn = gr.Button("♻️ Load / Reload model", variant="secondary")
        load_log = gr.Markdown()

    # Wire up actions
    load_btn.click(fn=init_model, inputs=[model_name, adapter_dir, dtype], outputs=load_log)

    # Auto-load once on app start with defaults
    demo.load(fn=init_model, inputs=[model_name, adapter_dir, dtype], outputs=load_log)

    run_btn.click(fn=run_ner, inputs=[input_text, max_tokens, max_seq_len, advanced], outputs=[highlighted, status_out])

    gr.Examples(
        examples=[
            [
                "Factura 2024-0098 emitida a Apple Retail Spain, S.L.U., Calle Príncipe de Vergara 112, 28002 Madrid. "
                "Contacto: john.doe@example.com, +34 600 123 456.",
                CFG.default_max_input_tokens,
                CFG.default_max_seq_len,
            ],
            [
                "El contrato entre ACME Corporation y Juan Pérez establece un pago de 12.500 € el 15/09/2025 en Valencia.",
                CFG.default_max_input_tokens,
                CFG.default_max_seq_len,
            ],
        ],
        inputs=[input_text, max_tokens, max_seq_len, advanced],
        label="Quick examples",
    )

if __name__ == "__main__":
    demo.queue(max_size=64).launch()
