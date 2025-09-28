import replicate
import dotenv
import json
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


dotenv.load_dotenv()

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()


def call_llm(prompt: str):
    output = replicate.run(
        "openai/gpt-5-nano",
        input={
            "prompt": prompt,
            "messages": [],
            "verbosity": "high",
            "image_input": [],
            "system_prompt": system_prompt,
            "reasoning_effort": "low",
            "max_completion_tokens": 4096
        },
    )
    output = "".join(output).strip()
    return output


all_data = json.load(open("wikipedia_abstracts_10k.json", "r", encoding="utf-8"))
all_data = [data["text"] for data in all_data]

os.makedirs("wikipedia_results", exist_ok=True)
already_processed_files = sorted(glob.glob("wikipedia_results/*.json"))
already_processed = {
    int(os.path.basename(file).rsplit("_", 1)[1].split(".")[0])
    for file in already_processed_files
}

max_workers = 32
indices_and_texts = [
    (idx, data)
    for idx, data in enumerate(all_data)
    if idx not in already_processed
]

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_task = {
        executor.submit(call_llm, data): (idx, data)
        for idx, data in indices_and_texts
    }
    
    with tqdm(total=len(future_to_task), desc="Processing Wikipedia abstracts") as pbar:
        for future in as_completed(future_to_task):
            idx, data = future_to_task[future]
            try:
                llm_output = future.result()
                result = {"input": data, "output": llm_output}
                with open(f"wikipedia_results/wikipedia_results_{idx}.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
            pbar.update(1)
