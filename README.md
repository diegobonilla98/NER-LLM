# NER-LLM — Label-Free NER with Lightweight LLMs

![Project Overview Placeholder](docs/images/project-overview.png)

NER-LLM is an end-to-end framework for **truly zero-shot, unconditioned named entity recognition**. Instead of hand-curated tag sets or token-level annotations, we generate a massive synthetic corpus with large models, then fine-tune a compact LLM (Qwen 0.6B) using LoRA/DoRA adapters to make it output clean JSON entity maps. The result is a deployable NER system that scales to any domain, supports multilingual inputs, and runs comfortably on a single consumer GPU.

- **Goal coverage:** ___% of benchmark suite (TODO)
- **Latency on RTX 4090:** ___ ms per 512 tokens (TODO)
- **Peak VRAM during inference:** ___ GB (TODO)

## Motivation
- Traditional NER stacks revolve around rigid ontologies, token classification heads, and labor-intensive annotation projects.
- QA-style prompting and instruction-tuned LLMs improved flexibility, yet still depend on manual prompt engineering and post-processing.
- Our approach pushes beyond past work by letting a small decoder-only LLM *learn the mapping itself*. We fine-tune it to emit only a JSON dictionary of `"surface_span": "entity_type"`, sidestepping CRFs, BIO tagging, and constrained decoding.

### Why a lightweight LLM beats classic encoders
- **Decoding-only objective:** Generation directly optimizes the final artifact (JSON spans), avoiding label alignment headaches and heuristics when entities overlap or repeat.
- **Scalability:** LoRA/DoRA adapters let us fine-tune billions of synthetic examples without full model churn, and we can swap base checkpoints quickly as better small LLMs arrive.
- **Composability:** The same model can simultaneously perform extraction, typing, and span normalization—tasks that traditionally require separate classifiers.
- **Deployment footprint:** Qwen 0.6B with adapters is <1 GB on disk and serves via GPU or CPU, eliminating heavyweight encoder stacks like RoBERTa or multilingual BERT.

## How This Differs from Prior Work
| Classic NER | Instruction-Following NER | NER-LLM |
|-------------|---------------------------|-------------|
| Token-level labels (BIO/IOB2) | LLMs infer labels with few-shot prompts | **Synthesized supervision + JSON-only decoding** |
| Fixed taxonomy, brittle to drift | Prompt drift, requires manual curation | **Unlimited label space** generated on demand |
| Costly annotation cycles | Demo-driven, hard to evaluate | **Automated data engine** generating >100k clean pairs |

Instead of collecting labeled corpora, we **generate them**. `create_dataset_llm.py`, `create_dataset_replicate_wiki.py`, and `personalize_dataset.py` orchestrate multiple synthetic sources (PII leak corpora, Wikipedia abstracts, multilingual datasets) and normalize them into consistent `input`/`output` JSON pairs. This provides richer supervision than earlier synthetic NER pipelines that only augmented labels inside fixed schemas.

## System Overview

![Pipeline Placeholder](docs/images/pipeline.png)

1. **Prompt Harvesting** (`create_prompts.py`): aggregate raw text from diverse corpora, cap length, and write a 100k prompt cache.
2. **Synthetic Labeling** (`create_dataset_llm.py`, `llm_utils.py`): stream prompts through GPT-5 Nano batches; capture outputs and reconcile retries.
3. **Wikipedia Augmentation** (`download_wikipedia.py`, `create_dataset_replicate_wiki.py`): expand coverage with encyclopedic facts in multiple languages.
4. **Dataset Assembly** (`save_entire_dataset.py`): merge all artifacts into `train.jsonl`/`val.jsonl` with balanced splits.
5. **Adapter Training** (`train_ner_lora_accelerate.py`): fine-tune Qwen/Qwen3-0.6B with DoRA-enabled LoRA adapters using Accelerate + DeepSpeed.
6. **Interactive Inference** (`app.py`, `inference.py`): launch a Gradio UI capable of loading different adapter checkpoints and visualizing spans.

## Data Engine in Detail
- **Batch-first design:** We rely on the OpenAI Batch API (`llm_utils.py`) to hit high throughput. Failed chunks are recursively resubmitted with adaptive batch sizes.
- **Consistent output schema:** `system_prompt.txt` enforces strict JSON to eliminate post-processing heuristics.
- **Coverage mixing:** Scripts in `personalize_dataset.py` ensure multilingual balance from `tner/multinerd`, while `create_dataset_replicate_wiki.py` injects factual and noisy prose to stress-test labeling.
- **Dataset consolidation:** `save_entire_dataset.py` removes duplicates, shuffles records, and materializes training/validation splits ready for Accelerate.

## Training Lightweight LLMs with LoRA/DoRA
- `train_ner_lora_accelerate.py` packs prompt/target pairs into 2k-token windows, applies DoRA-enhanced LoRA modules on attention and MLP projections, and leverages DeepSpeed ZeRO-2 when available.
- The script auto-saves best and last adapters to `ner_checkpoint_dorav2/qwen3-0p6b-ner-lora/`.
- Gradient accumulation + cosine LR schedule keep training stable on a single 24 GB GPU.

### Reproducing Training
```bash
accelerate launch train_ner_lora_accelerate.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_file train.jsonl \
  --val_file val.jsonl \
  --output_dir ner_checkpoint_dorav2/qwen3-0p6b-ner-lora \
  --epochs 2 --train_bs 2 --grad_accum 8 --bf16
```

## Inference & Demo UI
- `inference.py` loads the base model plus adapters, handles tokenizer quirks, and parses generated JSON into span offsets.
- `app.py` wraps everything in Gradio, showcasing highlighted entities, truncation warnings, and adapter switching (LoRA vs DoRA).

Run the UI locally:
```bash
python app.py
```

## Repository Highlights
- `create_dataset_llm.py` — batch synthetic generation pipeline.
- `save_entire_dataset.py` — merges batch artifacts and Wikipedia outputs into `train.jsonl`/`val.jsonl`.
- `train_ner_lora_accelerate.py` — Accelerate + DoRA fine-tuning script.
- `inference.py` — deterministic JSON-only decoding helper.
- `app.py` — Gradio interface for visual QA and demos.
- `system_prompt.txt` — canonical instruction ensuring strict JSON spans.

Everything else (temporary caches, intermediate artifacts) lives under `thinking_batch_artifacts/` and `wikipedia_results/` and is excluded from the core description.

## Getting Started
1. **Install dependencies** (excerpt):
   ```bash
   pip install -r requirements.txt  # TODO: author with exact list
   ```
2. **Download/base model & adapters:**
   ```bash
   python download_checkpoints.py
   ```
3. **Regenerate dataset (optional):**
   - Create prompts with `create_prompts.py`.
   - Batch label using `create_dataset_llm.py` (requires OpenAI + batch quota).
   - Augment with Wikipedia via `create_dataset_replicate_wiki.py`.
   - Consolidate using `save_entire_dataset.py`.
4. **Fine-tune adapters:** run `train_ner_lora_accelerate.py` with your hyperparameters.
5. **Serve the UI:** `python app.py` launches Gradio on `http://127.0.0.1:7860`.

## Evaluation (Work in Progress)
- Macro-F1 on synthetic validation: ___% (TODO)
- Human spot-check agreement: ___% (TODO)
- Cross-lingual recall (English/Spanish/German): ___% / ___% / ___% (TODO)

![Evaluation Placeholder](docs/images/eval-matrix.png)

## Roadmap
- Replace GPT-5 Nano labeling with open-source reasoning models.
- Quantize adapters for 4-bit CPU inference.
- Add automatic span merging metrics and benchmarking dashboards.

---

Questions or ideas? Open an issue or drop a pull request—this repo is intentionally modular so you can swap base models, prompts, or LoRA targets without rewriting the stack.

