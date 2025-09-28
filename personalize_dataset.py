import json
import random
from collections import defaultdict
import tqdm

MAX_TOKENS = 256
DATASET_SIZE = 60_000
seed = 42
rng = random.Random(seed)

def approx_token_count_from_tokens(tokens):
    char_len = sum(len(w) for w in tokens)
    if len(tokens) > 1:
        char_len += (len(tokens) - 1)
    return char_len // 4

def _uniform_language_schedule(dataset, out_size, seed=None):
    r = random.Random(seed)
    langs = sorted({d["language"] for d in dataset})
    if not langs:
        return []
    q, rmd = divmod(out_size, len(langs))
    schedule = [lang for lang in langs for _ in range(q)]
    if rmd:
        schedule += r.sample(langs, rmd)
    r.shuffle(schedule)
    return schedule

dataset = []
with open(r"F:\NER\multinerd_all_languages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

lang_to_indices = defaultdict(list)
for idx, ex in enumerate(dataset):
    lang_to_indices[ex["language"]].append(idx)

tok_len = [0] * len(dataset)
for i, ex in enumerate(dataset):
    tok_len[i] = approx_token_count_from_tokens(ex["tokens"])

lang_order = {}
lang_cursor = {}
for lang, idxs in lang_to_indices.items():
    idxs = idxs[:]
    rng.shuffle(idxs)
    lang_order[lang] = idxs
    lang_cursor[lang] = 0

schedule = _uniform_language_schedule(dataset, DATASET_SIZE, seed)
augmented = []
augmented_reserve = len(augmented)

pbar = tqdm.tqdm(total=DATASET_SIZE)

for lang in schedule:
    order = lang_order.get(lang)
    if not order:
        continue
    n = len(order)
    cur = lang_cursor[lang]

    base_idx = order[cur]
    cur = (cur + 1) % n
    lang_cursor[lang] = cur

    base_tokens = list(dataset[base_idx]["tokens"])
    base_tags   = list(dataset[base_idx]["tags"])
    total_tok   = tok_len[base_idx]

    if total_tok < MAX_TOKENS and n > 1:
        steps = 0
        idx_ptr = cur
        to_fill = MAX_TOKENS - total_tok
        while steps < n - 1 and to_fill > 0:
            j = order[idx_ptr]
            if j != base_idx:
                cand_len = tok_len[j]
                if cand_len <= to_fill:
                    # append
                    base_tokens.extend(dataset[j]["tokens"])
                    base_tags.extend(dataset[j]["tags"])
                    total_tok += cand_len
                    to_fill -= cand_len
            steps += 1
            idx_ptr = (idx_ptr + 1) % n
    pbar.update(1)

    output_tokens = []
    for token, tag in zip(base_tokens, base_tags):
        if tag == 0:
            output_tokens.append(token)
        else:
            output_tokens.append(f"<{tag}>{token}</{tag}>")

    augmented.append({
        "tokens": base_tokens,
        "text": " ".join(base_tokens),
        "output_tokens": output_tokens,
        "output_text": " ".join(output_tokens),
        "tags": base_tags,
        "language": lang
    })

with open(f"multinerd_all_languages_{MAX_TOKENS}.jsonl", "w", encoding="utf-8") as f:
    for ex in augmented:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
