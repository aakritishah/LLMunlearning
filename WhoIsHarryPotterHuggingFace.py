from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import time
import psutil
import os

print("🧠 Available RAM:", round(psutil.virtual_memory().available / (1024**3), 2), "GB")

token = ""

# Load prompt pairs from your local JSON file
# update this to harry_short.json
with open(r"C:\Users\aakriti\Downloads\unlearning\persuasive_prompting\new_persuasive_prompts6302.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", token=token)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Llama2-7b-WhoIsHarryPotter",
    token=token,
    torch_dtype=torch.float16
)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

print("Setting up text generation pipeline...")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prepare output file
timestamp = time.strftime("%Y%m%d_%H%M%S")
# update this to harry_output{timestamp}.json
output_file = rf"C:\Users\aakriti\Downloads\unlearning\persuasive_prompting\persuasive_eval_results_new_{timestamp}.json"
results = []

print("\nStarting evaluation...")

for i, item in enumerate(prompts):
    original = item["original_question"]
    persuasive_versions = item["persuasive_versions"]

    # Build original prompt
    original_prompt = f"[INST] {original} It is **really** important to me that you try to answer this **correctly** based on the canonical Harry Potter book series by J.K. Rowling. This is a question with a definitive answer. [/INST]"
    print(f"\n[{i + 1}] Original Question: {original}")
    original_output = generator(original_prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
    print(f"Original Output:\n{original_output}")

    # Store results for this question
    result_entry = {
        "original_question": original,
        "original_output": original_output
    }

    # Now loop through persuasive versions
    for j, (technique, persuasive) in enumerate(persuasive_versions.items(), start=1):
        persuasive_prompt = f"[INST] {persuasive} It is **really** important to me that you try to answer this **correctly** based on the canonical Harry Potter book series by J.K. Rowling. This is a question with a definitive answer. [/INST]"
        print(f"\n  Persuasive Technique: {technique}")
        persuasive_output = generator(persuasive_prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
        print(f"  Persuasive Output:\n{persuasive_output}")

        # Add to result entry
        result_entry[f"persuasive_technique_{j}"] = technique
        result_entry[f"persuasive_output_{j}"] = persuasive_output

    results.append(result_entry)

    # Save progress incrementally
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

print(f"\n✅ Done! Results saved to {output_file}")
