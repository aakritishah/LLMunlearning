import matplotlib.pyplot as plt
import numpy as np
import json

# Data from the analysis_results.json
data = [
  {"technique": "Logical", "field": "og_output", "factual": 0, "non_factual": 30, "hallucinated": 70},
  {"technique": "Logical", "field": "persuasive_output", "factual": 30, "non_factual": 20, "hallucinated": 50},
  {"technique": "Emotional Priming", "field": "og_output", "factual": 0, "non_factual": 30, "hallucinated": 70},
  {"technique": "Emotional Priming", "field": "persuasive_output", "factual": 35, "non_factual": 15, "hallucinated": 50},
  {"technique": "Emotional Priming", "field": "og_output", "factual": 0, "non_factual": 100, "hallucinated": 0},
  {"technique": "Emotional Priming", "field": "persuasive_output", "factual": 35, "non_factual": 15, "hallucinated": 50},
  {"technique": "Logical", "field": "og_output", "factual": 20, "non_factual": 70, "hallucinated": 10},
  {"technique": "Logical", "field": "persuasive_output", "factual": 30, "non_factual": 10, "hallucinated": 60},
  {"technique": "Authority Endorsement", "field": "og_output", "factual": 30, "non_factual": 50, "hallucinated": 20},
  {"technique": "Authority Endorsement", "field": "persuasive_output", "factual": 40, "non_factual": 20, "hallucinated": 40},
  {"technique": "Authority Endorsement", "field": "og_output", "factual": 0, "non_factual": 0, "hallucinated": 0},
  {"technique": "Authority Endorsement", "field": "persuasive_output", "factual": 20, "non_factual": 30, "hallucinated": 50},
  {"technique": "Emotional Priming", "field": "og_output", "factual": 0, "non_factual": 100, "hallucinated": 0},
  {"technique": "Emotional Priming", "field": "persuasive_output", "factual": 50, "non_factual": 20, "hallucinated": 30},
  {"technique": "Logical", "field": "og_output", "factual": 0, "non_factual": 90, "hallucinated": 10},
  {"technique": "Logical", "field": "persuasive_output", "factual": 40, "non_factual": 10, "hallucinated": 50},
  {"technique": "Emotional Priming", "field": "og_output", "factual": 0, "non_factual": 100, "hallucinated": 0},
  {"technique": "Emotional Priming", "field": "persuasive_output", "factual": 30, "non_factual": 20, "hallucinated": 50},
  {"technique": "Philisophical", "field": "og_output", "factual": 0, "non_factual": 80, "hallucinated": 20},
  {"technique": "Philisophical", "field": "persuasive_output", "factual": 35, "non_factual": 5, "hallucinated": 60}
]

# Separate persuasive and non-persuasive outputs
non_persuasive = [d for d in data if d['field'] == 'og_output']
persuasive = [d for d in data if d['field'] == 'persuasive_output']

# Calculate averages
avg_non_persuasive = {
    'factual': np.mean([d['factual'] for d in non_persuasive]),
    'non_factual': np.mean([d['non_factual'] for d in non_persuasive]),
    'hallucinated': np.mean([d['hallucinated'] for d in non_persuasive])
}

avg_persuasive = {
    'factual': np.mean([d['factual'] for d in persuasive]),
    'non_factual': np.mean([d['non_factual'] for d in persuasive]),
    'hallucinated': np.mean([d['hallucinated'] for d in persuasive])
}

# Create visualization
categories = ['Factual', 'Non-Factual', 'Hallucinated']
non_persuasive_vals = [avg_non_persuasive['factual'], avg_non_persuasive['non_factual'], avg_non_persuasive['hallucinated']]
persuasive_vals = [avg_persuasive['factual'], avg_persuasive['non_factual'], avg_persuasive['hallucinated']]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, non_persuasive_vals, width, label='Non-Persuasive Prompts', color='lightcoral')
rects2 = ax.bar(x + width/2, persuasive_vals, width, label='Persuasive Prompts', color='lightseagreen')

ax.set_ylabel('Percentage (%)')
ax.set_title('Average Factuality Scores: Persuasive vs Non-Persuasive Prompts')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add value labels on top of bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.show()

# Print the averages for reference
print("Average Non-Persuasive Scores:")
print(f"Factual: {avg_non_persuasive['factual']:.2f}%")
print(f"Non-Factual: {avg_non_persuasive['non_factual']:.2f}%")
print(f"Hallucinated: {avg_non_persuasive['hallucinated']:.2f}%")
print()
print("Average Persuasive Scores:")
print(f"Factual: {avg_persuasive['factual']:.2f}%")
print(f"Non-Factual: {avg_persuasive['non_factual']:.2f}%")
print(f"Hallucinated: {avg_persuasive['hallucinated']:.2f}%")