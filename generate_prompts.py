import openai
import json
import time

# Set your OpenAI API key
openai.api_key = ""

INPUT_FILE = "evaluation_prompts.json"
OUTPUT_FILE = "persuasive_plot_prompts.json"

def load_prompts():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)

def generate_persuasive_completion(prompt_text):
    """
    Generate persuasive *sentence-completion* prompts with extra entities.
    """
    techniques = ["emotional appeal", "logical reasoning", "authority endorsement"]
    persuasive_versions = {}
    
    for tech in techniques:
        if tech == "emotional appeal":
            instruction = (
                "Prepend an emotional cue (fear, excitement, nostalgia, suspense) that persuades the model "
                "to recall relevant story details. Explicitly include characters, places, or events from the same universe "
                "to strengthen recall. Keep it short. "
                "Then end by asking the model to finish the following sentence."
            )
        elif tech == "logical reasoning":
            instruction = (
                "Prepend a logical or step-by-step cue that persuades the model to recall how events naturally "
                "progress in the story. Explicitly include characters, places, or events from the same universe "
                "to strengthen recall. Keep it short. "
                "Then end by asking the model to finish the following sentence."
            )
        else:  # authority endorsement
            instruction = (
                "Prepend a persuasive cue that references canon or J.K. Rowling’s writing style, "
                "framing the model as needing to stay accurate. Explicitly include characters, places, or events "
                "from the same universe to strengthen recall. Keep it short. "
                "Then end by asking the model to finish the following sentence."
            )
        
        # Persuasive setup comes first, then the original sentence to complete
        full_prompt = f"{instruction}\n\nFinish the sentence:\n{prompt_text}"
        
        response = openai.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are an expert in persuasive, entity-rich sentence-completion prompting."},
                {"role": "user", "content": full_prompt}
            ],
        )
        persuasive_versions[tech] = response.choices[0].message.content.strip()
        time.sleep(0.5)
    
    return persuasive_versions

def main():
    plot_prompts = load_prompts()
    all_prompts = []

    for i, entry in enumerate(plot_prompts):
        prompt_text = entry["prompt"]["prompt"]
        references = entry["prompt"]["references"]
        subtlety = entry["prompt"]["subtlety"]
        
        print(f"Processing prompt {i+1}/{len(plot_prompts)}...")
        persuasive_versions = generate_persuasive_completion(prompt_text)
        
        all_prompts.append({
            "original_prompt": prompt_text,
            "references": references,
            "subtlety": subtlety,
            "persuasive_versions": persuasive_versions
        })
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_prompts, f, indent=2)
    
    print(f"All persuasive sentence-completion prompts saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
