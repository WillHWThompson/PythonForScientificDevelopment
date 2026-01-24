import matplotlib.pyplot as plt
import seaborn as sns

# --- Scientific Brand Palette ---
# A premium, high-contrast palette for professional publications
BRAND_COLORS = {
    "slate": "#2d3436",    # Dark background/text
    "emerald": "#00b894",  # Primary highlight
    "gold": "#fdcb6e",     # Secondary highlight (attention)
    "indigo": "#6c5ce7",   # Tertiary (divergent)
    "coral": "#ff7675",    # Error/Warning
    "ghost": "#f1f2f6"     # Light background
}

def set_scientific_style():
    """Configures Matplotlib and Seaborn for a premium scientific look."""
    # Build list for seaborn palette
    palette = [
        BRAND_COLORS["emerald"],
        BRAND_COLORS["gold"],
        BRAND_COLORS["indigo"],
        BRAND_COLORS["coral"],
        BRAND_COLORS["slate"]
    ]
    
    sns.set_theme(
        style="whitegrid", 
        palette=palette,
        font="sans-serif",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": BRAND_COLORS["slate"],
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "figure.figsize": (8, 5),
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "savefig.transparent": True
        }
    )
    
    # Also set raw matplotlib parameters for non-seaborn plots
    plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Arial"]
    plt.rcParams["axes.labelcolor"] = BRAND_COLORS["slate"]
    plt.rcParams["xtick.color"] = BRAND_COLORS["slate"]
    plt.rcParams["ytick.color"] = BRAND_COLORS["slate"]

if __name__ == "__main__":
    # Demo the palette
    set_scientific_style()
    sns.palplot(sns.color_palette())
    plt.title("Scientific Brand Palette")
    plt.show()
