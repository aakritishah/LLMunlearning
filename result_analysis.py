import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import json

def safe_extract(formula_entry, key=None):
    """
    Safely extract a numeric value from the formula entry.
    If it's a dict, use key to get value. If it's a number, return it directly.
    """
    if isinstance(formula_entry, dict):
        return formula_entry.get(key, 0) if key else 0
    elif isinstance(formula_entry, (int, float)):
        return formula_entry
    else:
        return 0

def parse_results_json(filename):
    """
    Parse the JSON results file and extract entanglement scores for each prompt.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for entry in data:
        row = {
            'prompt_text': entry.get('prompt', ''),
            'prompt_type': entry.get('prompt_type', 'unknown')
        }
        formulas = entry.get('formulas', {})

        row['formula1_edge_count_total_weight'] = safe_extract(formulas.get('formula1_edge_count_total_weight'), 'total_edge_weight')
        row['formula2_weighted_node_ratio'] = safe_extract(formulas.get('formula2_weighted_node_ratio'), 'ratio')
        row['formula3_avg_node_degree_entanglement'] = safe_extract(formulas.get('formula3_avg_node_degree_entanglement'), 'average_degree')
        row['formula4_subgraph_density'] = safe_extract(formulas.get('formula4_subgraph_density'), 'density')
        row['formula5_edge_weight_sum'] = safe_extract(formulas.get('formula5_edge_weight_sum'), 'total_edge_weight_sum')
        row['formula6_avg_edge_weight_sum'] = safe_extract(formulas.get('formula6_avg_edge_weight_sum'))
        row['formula7_mean_shortest_path'] = safe_extract(formulas.get('formula7_mean_shortest_path'))
        row['formula8_redundancy_ratio'] = safe_extract(formulas.get('formula8_redundancy_ratio'))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def analyze_entanglement_by_type(df):
    """
    Perform statistical analysis on entanglement scores by prompt type.
    """
    formula_cols = [col for col in df.columns if col.startswith('formula')]
    
    print("=" * 60)
    print("SUMMARY STATISTICS BY PROMPT TYPE")
    print("=" * 60)
    
    summary_stats = df.groupby('prompt_type')[formula_cols].agg(['mean', 'std', 'count'])
    print(summary_stats.round(3))
    
    # Encode prompt types (original=0, emotional=1, logical=2, authority=3)
    le = LabelEncoder()
    df['prompt_type_encoded'] = le.fit_transform(df['prompt_type'])
    
    results = {}
    
    for formula_col in formula_cols:
        X = df[['prompt_type_encoded']]
        y = df[formula_col].astype(float)
        
        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        corr, p_value = stats.pearsonr(df['prompt_type_encoded'], y)
        
        results[formula_col] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'correlation': corr,
            'p_value': p_value
        }
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"\n{formula_col.upper()}:")
        print(f"  Slope: {slope:.4f} {significance}")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} relationship")
    
    return results

def create_visualizations(df):
    """
    Create visualizations to show relationships between prompt types and entanglement scores.
    """
    formula_cols = [col for col in df.columns if col.startswith('formula')]
    plt.style.use('default')
    sns.set_palette("husl")
    
    n_formulas = len(formula_cols)
    n_cols = 3
    n_rows = (n_formulas + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, formula_col in enumerate(formula_cols):
        ax = axes[i]
        sns.boxplot(data=df, x='prompt_type', y=formula_col, ax=ax)
        ax.set_title(f'{formula_col.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
        
        # Trend line
        prompt_type_numeric = df['prompt_type'].map({
            'original': 0, 'emotional': 1, 'logical': 2, 'authority': 3
        })
        z = np.polyfit(prompt_type_numeric, df[formula_col], 1)
        p = np.poly1d(z)
        ax.plot([0,1,2,3], p([0,1,2,3]), "r--", alpha=0.8, linewidth=2)
    
    for i in range(len(formula_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('entanglement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    correlation_data = df[formula_cols + ['prompt_type_encoded']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix: Formulas vs Prompt Type')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main analysis function.
    """
    print("Parsing JSON results file...")
    df = parse_results_json('prompt_analysis_results.json')  # Updated to JSON
    
    print(f"Successfully parsed {len(df)} prompts")
    print("Prompt type distribution:")
    print(df['prompt_type'].value_counts())
    
    results = analyze_entanglement_by_type(df)
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    df.to_csv('entanglement_analysis_results.csv', index=False)
    print("\nResults exported to 'entanglement_analysis_results.csv'")
    
    return df, results

if __name__ == "__main__":
    df, results = main()
