import matplotlib.pyplot as plt
import seaborn as sns

# --- Modern Pastel Purple Palette ---
# A professional, clean palette with white backgrounds
BRAND_COLORS = {
    "text": "#2d3436",      # Deep charcoal text
    "purple": "#6c5ce7",    # Primary (Main)
    "mint": "#26de81",      # Accent (Complementary)
    "peach": "#fed330",     # Secondary Accent
    "lavender": "#a29bfe",  # Tertiary
    "bg": "#ffffff"         # Pure white background
}

def set_scientific_style():
    """Configures Matplotlib and Seaborn for a clean, modern pastel look."""
    # Build list for seaborn palette
    palette = [
        BRAND_COLORS["purple"],
        BRAND_COLORS["mint"],
        BRAND_COLORS["peach"],
        BRAND_COLORS["lavender"],
        BRAND_COLORS["text"]
    ]
    
    sns.set_theme(
        style="whitegrid", 
        palette=palette,
        font="sans-serif",
        rc={
            "axes.facecolor": BRAND_COLORS["bg"],
            "figure.facecolor": BRAND_COLORS["bg"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#dfe6e9", # Light gray borders
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
    plt.title("Modern Pastel Purple Palette")
    plt.show()
