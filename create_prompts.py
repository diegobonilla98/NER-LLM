import json
import pandas as pd
from tqdm import tqdm
from concatenate_tokens import concatenate_tokens
import random
import re

all_prompts = []

pii_files = [
    "0ce7f7a670f398a48fcbe1ac328a01adb48187cb7c2975c99d46d0e72176103e.jsonl",
    "8bd82d8eb08154343f0f351e4b33d10176ea72fe4b8e9c40e6e0d81b53bf6c92.jsonl",
    "32fc0f89232197378f397ac7edcf49b49228b3676452bbf688df2a573c3d5057.jsonl",
    "58a679746c53f5749fe3a573791ac1862973d8c524c9638fcb233a29041557e6.jsonl",
    "381ae8317a40e9c870ddd741dd162a50135746da393f3da37731bcacc6c03f9c.jsonl",
    "fd29913cee752939e01a39b7ac70973a570bc8b5e140ff030df5e4c8043dfa6c.jsonl"
]

for file in pii_files:
    with open(rf"F:\NER\{file}", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            all_prompts.append(data["source_text"])

with open(r"F:\NER\znf_nb_general_topic_1k_0220_mixtral.json", "r", encoding="utf-8") as file:
    all_data = json.load(file)
for data in tqdm(all_data, desc="Concatenating tokens"):
    all_prompts.append(data["full_text"])

with open(r"F:\NER\lzc_noise_data_2000_0214_augmented.json", "r", encoding="utf-8") as file:
    all_data = json.load(file)
for data in tqdm(all_data, desc="Concatenating tokens"):
    all_prompts.append(data["full_text"])

with open(r"F:\NER\1english_openpii_8k.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        all_prompts.append(data["source_text"])

with open(r"F:\NER\train.json", "r", encoding="utf-8") as file:
    all_data = json.load(file)
for data in tqdm(all_data, desc="Concatenating tokens"):
    all_prompts.append(data["full_text"].replace("•", ""))

with open(r"F:\NER\multinerd_all_languages_256.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        all_data.append(data)
for data in tqdm(all_data, desc="Concatenating tokens"):
    all_prompts.append(concatenate_tokens(data["tokens"]))

all_prompts_clean = []
for prompt in tqdm(all_prompts, desc="Cleaning prompts"):
    cleaned = re.sub(r'\s+', ' ', prompt)
    num_tokens_approx = len(cleaned) // 4
    if num_tokens_approx > 256:
        # truncate the string to the last space before the 256th token count
        words = cleaned.split(" ")
        new_text = ""
        for i in range(len(words)):
            new_text += words[i]
            if len(new_text) // 4 > 256:
                break
            new_text += " "
        cleaned = new_text.strip()
    all_prompts_clean.append(cleaned)

N = 100_000
all_prompts = random.sample(all_prompts, N)

# save pickle
import pickle
with open(r"F:\NER\all_prompts_clean.pkl", "wb") as f:
    pickle.dump(all_prompts_clean, f)
