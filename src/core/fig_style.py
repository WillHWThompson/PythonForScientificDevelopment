import matplotlib.pyplot as plt
import seaborn as sns

# --- Professional Scientific Palette (Tol Vibrant) ---
# A colorblind-friendly, professional palette for complex visualizations
BRAND_COLORS = {
    "text": "#2d3436",
    "bg": "#ffffff",
    "palette": [
        "#332288", "#88CCEE", "#44AA99", "#117733", 
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
    ]
}

def set_scientific_style():
    """Configures Matplotlib and Seaborn for a professional, colorblind-friendly look."""
    sns.set_theme(
        style="whitegrid", 
        palette=BRAND_COLORS["palette"],
        font="sans-serif",
        rc={
            "axes.facecolor": BRAND_COLORS["bg"],
            "figure.facecolor": BRAND_COLORS["bg"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#dfe6e9", 
            "grid.color": "#f1f2f6",
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            "figure.figsize": (8, 5),
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": BRAND_COLORS["text"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": BRAND_COLORS["text"],
            "ytick.color": BRAND_COLORS["text"],
            "legend.frameon": False,
            "savefig.transparent": False,
            "savefig.facecolor": BRAND_COLORS["bg"]
        }
    )
    
    # Also set raw matplotlib parameters
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Arial"]

if __name__ == "__main__":
    # Demo the palette
    set_scientific_style()
    sns.palplot(sns.color_palette())
    plt.title("Professional Scientific Palette")
    plt.show()
