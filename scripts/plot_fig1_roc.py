import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse

# Add src to path so we can import fig_style
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core.fig_style import set_scientific_style, BRAND_COLORS

def plot_faceted_roc(input_files, output_file):
    set_scientific_style()
    
    all_data = []
    for f in input_files:
        # Expecting path like results/nlp_baseline/adapter_dim~128/learning_rate~0.0001/test_roc_curves.parquet
        p = Path(f)
        # Extract params from parent directories
        params = {}
        for part in p.parts:
            if "~" in part:
                k, v = part.split("~")
                params[k] = v
        
        df = pd.read_parquet(f)
        if "fpr" not in df.columns or "tpr" not in df.columns or "class" not in df.columns:
            print(f"Skipping {f}: missing required columns (fpr, tpr, class)")
            continue
            
        for k, v in params.items():
            df[k] = v
            
        all_data.append(df)
    
    if not all_data:
        print("No valid data found to plot.")
        return

    df_combined = pd.concat(all_data)
    
    # Ensure categorical sorting for adapter_dim
    if "adapter_dim" in df_combined.columns:
        df_combined["adapter_dim"] = pd.to_numeric(df_combined["adapter_dim"])
        df_combined = df_combined.sort_values("adapter_dim")
    
    # Create the faceted plot
    g = sns.FacetGrid(
        df_combined, 
        col="class", 
        col_wrap=2, 
        height=4, 
        aspect=1.2,
        hue="adapter_dim",
        palette=BRAND_COLORS["palette"]
    )
    
    # Plot ROC curves in each facet
    g.map(sns.lineplot, "fpr", "tpr", linewidth=2, alpha=0.8)
    
    # Add diagonal baseline and style each facet
    def add_baseline(**kwargs):
        plt.plot([0, 1], [0, 1], linestyle="--", color="#636e72", alpha=0.5)
        plt.grid(True, linestyle="--", alpha=0.3)

    g.map(add_baseline)
    
    # Layout and labels
    g.set_axis_labels("False Positive Rate", "True Positive Rate")
    g.set_titles(col_template="{col_name}", weight="bold")
    g.add_legend(title="Adapter Dim", frameon=False)
    
    plt.subplots_adjust(top=0.9)
    g.figure.suptitle("Figure 1: Class-Specific ROC Curves by Adapter Dimension", fontweight="bold", fontsize=16)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Faceted figure saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="List of parquet files")
    parser.add_argument("--output", required=True, help="Output image file")
    args = parser.parse_args()
    
    plot_faceted_roc(args.inputs, args.output)
