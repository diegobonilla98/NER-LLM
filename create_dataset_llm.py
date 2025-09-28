import json
import pickle
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Iterable, Set, Tuple

from llm_utils import (
    ThinkBatchConfig,
    submit_thinking_batch,
    wait_batch,
    download_batch_outputs,
    parse_batch_outputs,
)

from tqdm.auto import tqdm


# ---------------------------
# Configuration
# ---------------------------
PROMPTS_PKL_PATH = r"F:\NER\all_prompts_clean.pkl"
OUTPUT_DIR = Path(r"F:\NER\llm_ner")

# Tune to fit provider batch limits; 5k keeps files reasonable and cuts queue time
BATCH_SIZE = 500
MAX_RETRIES = 3
POLL_SECONDS = 20.0
MAX_CONSECUTIVE_ERROR_CHUNKS = 5
MIN_FALLBACK_BATCH_SIZE = 100
FALLBACK_BATCH_SIZE_CAP = 1000


# ---------------------------
# Utilities
# ---------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def atomic_write_json(path: Path, obj: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


def load_prompts(pkl_path: str) -> List[str]:
    with open(pkl_path, "rb") as f:
        prompts = pickle.load(f)
    if not isinstance(prompts, list):
        raise TypeError("Expected a list of prompts from pickle file")
    return prompts


def existing_indices(out_dir: Path) -> Set[int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    found: Set[int] = set()
    for p in out_dir.glob("*.json"):
        m = re.match(r"(\d+)", p.stem)
        if m:
            try:
                found.add(int(m.group(1)))
            except ValueError:
                continue
    return found


def chunked(items: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_batch_for_indices(
    all_prompts: List[str],
    idxs: List[int],
    system_prompt: str,
    work_root: Path,
    model: str = "gpt-5-nano",
) -> Tuple[int, int]:
    """Submit a batch for a set of indices and save results individually.

    Returns (saved_count, total_requested).
    """
    if not idxs:
        return 0, 0

    prompts = [all_prompts[i] for i in idxs]
    start_idx, end_idx = idxs[0], idxs[-1]

    attempts = 0
    while attempts < MAX_RETRIES:
        attempts += 1
        try:
            batch_workdir = work_root / f"batch_{start_idx:06d}_{end_idx:06d}_try{attempts}"
            batch_workdir.mkdir(parents=True, exist_ok=True)
            # Persist exact indices used for this batch to support precise resume
            try:
                (batch_workdir / "indices.json").write_text(
                    json.dumps({"idxs": idxs}, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                logging.exception(f"Failed to persist indices for {batch_workdir}")
            cfg = ThinkBatchConfig(
                model=model,
                system_prompt=system_prompt,
                prompts=prompts,
                max_output_tokens=None,
                temperature=None,
                reasoning_effort="low",
                verbosity="medium",
                include=None,
                tools=None,
                store=False,
                metadata=None,
                completion_window="24h",
                job_name=f"ner-batch-{start_idx:06d}-{end_idx:06d}",
                workdir=batch_workdir,
            )

            batch_info = submit_thinking_batch(cfg)
            batch_id = batch_info.get("id") or batch_info.get("data", {}).get("id")
            logging.info(
                f"Submitted batch {batch_id} for indices [{start_idx}, {end_idx}] with {len(prompts)} items."
            )

            final_info = wait_batch(batch_id, poll_sec=POLL_SECONDS)
            logging.info(
                f"Batch {batch_id} finished with status {final_info.get('status')}"
            )

            status = final_info.get("status")
            if status != "completed":
                logging.error(
                    f"Batch {batch_id} ended with status '{status}'. Skipping this chunk without outputs."
                )
                return 0, len(idxs)

            paths = download_batch_outputs(final_info, out_dir=batch_workdir / "downloads")
            outputs_path = paths.get("outputs")
            if not outputs_path or not outputs_path.exists():
                logging.error(
                    f"No outputs.jsonl for batch {batch_id}. Errors file: {paths.get('errors')}"
                )
                return 0, len(idxs)

            parsed = parse_batch_outputs(outputs_path)  # {custom_id: {...}}
            saved = 0
            # Map item position in this batch -> global index
            for j, global_idx in enumerate(idxs):
                cid = f"item-{j:06d}"
                row = parsed.get(cid)
                if not row:
                    continue
                text = row.get("text")
                if not text:
                    continue
                record = {"input_prompt": prompts[j], "output_prompt": text}
                out_path = OUTPUT_DIR / f"{global_idx:06d}.json"
                try:
                    atomic_write_json(out_path, record)
                    saved += 1
                    if saved % 100 == 0:
                        logging.info(
                            f"Saved {saved}/{len(idxs)} results for batch {batch_id}"
                        )
                except Exception:
                    logging.exception(
                        f"Failed writing result for index {global_idx} to {out_path}"
                    )

            return saved, len(idxs)

        except Exception:
            logging.exception(
                f"Batch for indices [{start_idx}, {end_idx}] attempt {attempts} failed"
            )
            # Backoff between attempts; cap wait to 5 minutes
            time.sleep(min(60 * attempts, 300))

    # After retries, return whatever we managed to save (could be zero)
    return 0, len(idxs)


def _parse_batch_dir_indices(batch_dir: Path) -> List[int]:
    idx_file = batch_dir / "indices.json"
    if idx_file.exists():
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("idxs"), list):
                return [int(i) for i in data["idxs"]]
        except Exception:
            logging.exception(f"Failed reading indices.json in {batch_dir}")
    # Fallback to contiguous range based on directory name
    m = re.match(r"batch_(\d{6})_(\d{6})_try(\d+)$", batch_dir.name)
    if not m:
        return []
    start_idx = int(m.group(1))
    end_idx = int(m.group(2))
    if end_idx < start_idx:
        return []
    return list(range(start_idx, end_idx + 1))


def scan_completed_batches(work_root: Path) -> List[Tuple[Path, Path]]:
    """Return list of (batch_dir, outputs_path) that have outputs.jsonl present."""
    out: List[Tuple[Path, Path]] = []
    if not work_root.exists():
        return out
    for outputs_path in work_root.glob("batch_*_*_try*/downloads/outputs.jsonl"):
        batch_dir = outputs_path.parent.parent
        out.append((batch_dir, outputs_path))
    return out


def reconcile_from_artifacts(
    all_prompts: List[str], work_root: Path, out_dir: Path
) -> int:
    """Parse any existing outputs.jsonl artifacts and write missing output files.

    Returns the number of new items written.
    """
    written = 0
    completed = scan_completed_batches(work_root)
    if not completed:
        return 0
    logging.info(f"Found {len(completed)} completed batch artifacts to reconcile.")
    for batch_dir, outputs_path in completed:
        try:
            idxs = _parse_batch_dir_indices(batch_dir)
            if not idxs:
                logging.warning(f"Cannot infer indices for {batch_dir}; skipping.")
                continue
            parsed = parse_batch_outputs(outputs_path)
            for j, global_idx in enumerate(idxs):
                out_path = out_dir / f"{global_idx:06d}.json"
                if out_path.exists():
                    continue
                row = parsed.get(f"item-{j:06d}")
                if not row:
                    continue
                text = row.get("text")
                if not text:
                    continue
                record = {"input_prompt": all_prompts[global_idx], "output_prompt": text}
                try:
                    atomic_write_json(out_path, record)
                    written += 1
                    if written % 1000 == 0:
                        logging.info(
                            f"Reconciled {written} items so far from artifacts..."
                        )
                except Exception:
                    logging.exception(
                        f"Failed writing reconciled result for index {global_idx}"
                    )
        except Exception:
            logging.exception(f"Failed reconciling artifact at {batch_dir}")
    logging.info(f"Reconciled total {written} items from existing artifacts.")
    return written


def main() -> None:
    setup_logging()

    try:
        prompts = load_prompts(PROMPTS_PKL_PATH)
    except Exception:
        logging.exception(f"Failed loading prompts from {PROMPTS_PKL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        logging.exception("Failed reading system_prompt.txt")
        return

    work_root = Path("thinking_batch_artifacts") / "ner_batches"

    # First, reconcile any completed artifacts to avoid re-submitting already processed items
    try:
        reconcile_from_artifacts(prompts, work_root, OUTPUT_DIR)
    except Exception:
        logging.exception("Reconciliation from artifacts failed; proceeding with remaining work.")

    done = existing_indices(OUTPUT_DIR)
    total = len(prompts)
    remaining = [i for i in range(total) if i not in done]

    logging.info(
        f"Total prompts: {total}. Already saved: {len(done)}. Remaining: {len(remaining)}."
    )

    if not remaining:
        logging.info("Nothing to process. Exiting.")
        return

    grand_saved = 0
    consecutive_error_chunks = 0
    with tqdm(total=total, initial=len(done), desc="Dataset creation", unit="prompt") as pbar:
        for idx_chunk in chunked(remaining, BATCH_SIZE):
            try:
                saved, requested = run_batch_for_indices(
                    prompts, idx_chunk, system_prompt, work_root
                )
                grand_saved += saved
                pbar.update(saved)
                pbar.set_postfix({
                    "last_saved": saved,
                    "cum_saved": grand_saved,
                    "err_seq": consecutive_error_chunks,
                })
                logging.info(
                    f"Chunk [{idx_chunk[0]}, {idx_chunk[-1]}]: saved {saved}/{requested}. Cumulative saved: {grand_saved}"
                )

                if saved == 0 and requested > 1:
                    # Adaptive fallback: split this chunk into smaller pieces and retry
                    fallback_size = max(
                        MIN_FALLBACK_BATCH_SIZE,
                        min(FALLBACK_BATCH_SIZE_CAP, max(1, len(idx_chunk) // 2)),
                    )
                    logging.warning(
                        f"Primary batch saved 0 for chunk [{idx_chunk[0]}, {idx_chunk[-1]}]. "
                        f"Retrying with smaller sub-batches of size {fallback_size}."
                    )
                    sub_saved_total = 0
                    for sub_chunk in chunked(idx_chunk, fallback_size):
                        sub_saved, sub_requested = run_batch_for_indices(
                            prompts, sub_chunk, system_prompt, work_root
                        )
                        sub_saved_total += sub_saved
                        grand_saved += sub_saved
                        pbar.update(sub_saved)
                        pbar.set_postfix({
                            "last_saved": sub_saved,
                            "cum_saved": grand_saved,
                            "err_seq": consecutive_error_chunks,
                        })
                        logging.info(
                            f"Sub-chunk [{sub_chunk[0]}, {sub_chunk[-1]}]: saved {sub_saved}/{sub_requested}. "
                            f"Cumulative saved: {grand_saved}"
                        )

                    if sub_saved_total == 0:
                        consecutive_error_chunks += 1
                    else:
                        consecutive_error_chunks = 0
                elif saved == 0:
                    consecutive_error_chunks += 1
                else:
                    consecutive_error_chunks = 0

                if consecutive_error_chunks >= MAX_CONSECUTIVE_ERROR_CHUNKS:
                    logging.error(
                        f"Aborting run: encountered {consecutive_error_chunks} consecutive failing chunks."
                    )
                    break
            except Exception:
                logging.exception(
                    f"Unhandled error processing chunk [{idx_chunk[0]}, {idx_chunk[-1]}]"
                )
                consecutive_error_chunks += 1
                pbar.set_postfix({
                    "last_saved": 0,
                    "cum_saved": grand_saved,
                    "err_seq": consecutive_error_chunks,
                })
                if consecutive_error_chunks >= MAX_CONSECUTIVE_ERROR_CHUNKS:
                    logging.error(
                        f"Aborting run: encountered {consecutive_error_chunks} consecutive failing chunks."
                    )
                    break

    logging.info(
        f"Done. Saved {grand_saved} results. Output directory: {OUTPUT_DIR}" 
    )


if __name__ == "__main__":
    main()
