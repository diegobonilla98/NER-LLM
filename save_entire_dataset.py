import json
import glob
import os
import random


all_results = []
wiki_results = glob.glob("wikipedia_results/*.json")
dataset_results = glob.glob("thinking_batch_artifacts/ner_batches/*/downloads/outputs.jsonl")

for dataset_result in dataset_results:
    responses_file = dataset_result
    inputs_file = dataset_result.replace("downloads\\outputs.jsonl", "requests.jsonl")

    # Load responses
    with open(responses_file, "r", encoding="utf-8") as f:
        llm_response = [json.loads(line.strip()) for line in f]

    # Load inputs
    with open(inputs_file, "r", encoding="utf-8") as f:
        llm_input = [json.loads(line.strip()) for line in f]

    for resp, inp in zip(llm_response, llm_input):
        # --- Extract assistant output ---
        out_text = next(
            content["text"]
            for output in resp["response"]["body"]["output"]
            if output["type"] == "message"
            for content in output.get("content", [])
            if content.get("type") == "output_text"
        )

        # --- Extract user input ---
        in_text = next(
            content["text"]
            for message in inp["body"]["input"]
            if message["role"] == "user"
            for content in message.get("content", [])
            if content.get("type") == "input_text"
        )

        all_results.append({"input": in_text, "output": out_text})


for wiki_result in wiki_results:
    with open(wiki_result, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_results.append(data)

print(len(all_results))
split = 0.8
random.shuffle(all_results)
train_split = all_results[:int(len(all_results) * split)]
val_split = all_results[int(len(all_results) * split):]

with open("train.jsonl", "w", encoding="utf-8") as f:
    for data in train_split:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

with open("val.jsonl", "w", encoding="utf-8") as f:
    for data in val_split:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
