import argparse
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe

# Style setup
BACKGROUND = "#0f1117"
CARD_BG = "#1a1d27"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2d3a"
ACCENT_BLUE = "#5b9bd5"
ACCENT_GREEN = "#6bcb77"
ACCENT_YELLOW = "#ffd93d"
ACCENT_ORANGE = "#ff8c42"
ACCENT_RED = "#ff6b6b"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "font.family": "sans-serif",
    "font.size": 11,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

GLOW = [pe.withStroke(linewidth=3, foreground=BACKGROUND)]


# Applies consistent visual style (grid, spines, labels) to a matplotlib axis
def _style_ax(ax, ylabel=None, xlabel=None):
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight="medium")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight="medium")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))


# Helpers
# Counts files with a given extension inside a directory
def count_files(directory: Path, extension: str) -> int:
    return len(list(directory.glob(f"*.{extension}")))


# Reads .txt files from the given directories and extracts the quality score and status from each
def parse_quality_from_files(*directories: Path) -> list[dict]:
    records = []
    for directory in directories:
        for file_path in directory.glob("*.txt"):
            score = None
            status = None
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("QUALITY_SCORE:"):
                        try:
                            score = float(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    elif line.startswith("QUALITY_STATUS:"):
                        status = line.split(":", 1)[1].strip()
                    elif line.startswith("CONTENIDO:"):
                        break
            if score is not None and status is not None:
                records.append({"score": score, "status": status})
    return records


# Chart 1: Web Scraping overview
# Generates a bar and donut chart summarizing successfully scraped files vs failed ones
def plot_extracted_files(sitemap_total: int, html_count: int, output_path: Path):
    failed = sitemap_total - html_count
    success_pct = html_count / sitemap_total * 100
    failed_pct = failed / sitemap_total * 100

    labels = ["Total URLs\nen sitemap", "Descargados\nexitosamente", "Fallidos /\nRepetidos"]
    values = [sitemap_total, html_count, failed]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED]

    fig, (ax_bar, ax_donut) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Bar chart
    bars = ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.55,
                       edgecolor=CARD_BG, linewidth=1.5, zorder=3)
    for bar, val in zip(bars, values[::-1]):
        ax_bar.text(
            val + sitemap_total * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=13, fontweight="bold", color=TEXT_COLOR,
            path_effects=GLOW,
        )
    ax_bar.set_xlim(0, sitemap_total * 1.18)
    ax_bar.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax_bar.grid(axis="x", linewidth=0.6)
    ax_bar.grid(axis="y", visible=False)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(GRID_COLOR)
    ax_bar.spines["bottom"].set_color(GRID_COLOR)
    ax_bar.tick_params(length=0)

    # Donut chart
    wedges, _ = ax_donut.pie(
        [html_count, failed],
        colors=[ACCENT_GREEN, ACCENT_RED],
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": CARD_BG, "linewidth": 2},
    )
    ax_donut.text(0, 0.06, f"{success_pct:.1f}%", ha="center", va="center",
                  fontsize=22, fontweight="bold", color=ACCENT_GREEN)
    ax_donut.text(0, -0.16, "exitoso", ha="center", va="center",
                  fontsize=10, color="#999999")

    ax_donut.legend(
        [f"Exitosos  ({html_count:,})", f"Fallidos   ({failed:,})"],
        loc="lower center", fontsize=9, frameon=False,
        labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle("Archivos extraidos por Web Scraping",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


# Chart 2: Quality distribution
# Generates a bar chart showing the distribution of files by quality level (HIGH, MEDIUM, LOW, DEFECTIVE)
def plot_quality_distribution(records: list[dict], output_path: Path):
    order = ["HIGH", "MEDIUM", "LOW", "DEFECTIVE"]
    color_map = {
        "HIGH": ACCENT_GREEN, "MEDIUM": ACCENT_YELLOW,
        "LOW": ACCENT_ORANGE, "DEFECTIVE": ACCENT_RED,
    }
    counts = Counter(r["status"] for r in records)
    total = sum(counts.values())
    statuses = [s for s in order if s in counts]
    values = [counts[s] for s in statuses]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        statuses, values,
        color=[color_map[s] for s in statuses],
        width=0.55, edgecolor=CARD_BG, linewidth=1.5, zorder=3,
    )
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
            f"{val:,}", ha="center", fontsize=13, fontweight="bold",
            color=TEXT_COLOR, path_effects=GLOW,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.065,
            f"({pct:.1f}%)", ha="center", fontsize=9, color="#999999",
        )
    _style_ax(ax, ylabel="Cantidad de archivos", xlabel="Nivel de calidad")
    ax.set_ylim(0, max(values) * 1.18)

    fig.suptitle("Distribucion de archivos por calidad de limpieza",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


# Chart 3: Usable vs unusable
# Generates a bar and donut chart comparing usable files (score >= 70) vs unusable ones
def plot_usable_vs_unusable(usable: int, unusable: int, output_path: Path):
    total = usable + unusable
    usable_pct = usable / total * 100
    unusable_pct = unusable / total * 100

    fig, (ax_bar, ax_donut) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # Bar chart
    labels = ["Usables\n(score >= 70)", "No usables\n(score < 70)"]
    values = [usable, unusable]
    colors = [ACCENT_GREEN, ACCENT_RED]

    bars = ax_bar.bar(labels, values, color=colors, width=0.5,
                      edgecolor=CARD_BG, linewidth=1.5, zorder=3)
    for bar, val, pct in zip(bars, values, [usable_pct, unusable_pct]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
            f"{val:,}", ha="center", fontsize=14, fontweight="bold",
            color=TEXT_COLOR, path_effects=GLOW,
        )
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.065,
            f"({pct:.1f}%)", ha="center", fontsize=10, color="#999999",
        )
    _style_ax(ax_bar, ylabel="Cantidad de archivos")
    ax_bar.set_ylim(0, max(values) * 1.18)

    # Donut chart
    wedges, _ = ax_donut.pie(
        values, colors=colors, startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": CARD_BG, "linewidth": 2},
    )
    ax_donut.text(0, 0.06, f"{usable_pct:.1f}%", ha="center", va="center",
                  fontsize=24, fontweight="bold", color=ACCENT_GREEN)
    ax_donut.text(0, -0.18, "usables", ha="center", va="center",
                  fontsize=10, color="#999999")

    ax_donut.legend(
        [f"Usables       ({usable:,})", f"No usables  ({unusable:,})"],
        loc="lower center", fontsize=9, frameon=False,
        labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle("Archivos usables vs no usables",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


# Main
# Parses command-line arguments and runs the generation of all three charts
def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics charts from scraped and cleaned data"
    )
    parser.add_argument(
        "--html-dir",
        default="data/Laws",
        help="Directory with raw scraped HTML files (default: data/Laws)",
    )
    parser.add_argument(
        "--cleaned-dir",
        default="dataCleaned/Laws",
        help="Directory with usable cleaned TXT files (default: dataCleaned/Laws)",
    )
    parser.add_argument(
        "--unusable-dir",
        default="dataCleaned/unusable_files",
        help="Directory with unusable TXT files (default: dataCleaned/unusable_files)",
    )
    parser.add_argument(
        "--sitemap-total",
        type=int,
        default=11680,
        help="Total URLs found in the sitemap (default: 11680)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="Scripts/DatasetStats/output",
        help="Directory to save generated charts (default: Scripts/DatasetStats/output)",
    )
    args = parser.parse_args()

    html_dir = Path(args.html_dir)
    cleaned_dir = Path(args.cleaned_dir)
    unusable_dir = Path(args.unusable_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count raw HTML files
    html_count = count_files(html_dir, "html")
    print(f"Archivos HTML extraidos: {html_count}")

    # Parse quality data from cleaned + unusable files
    print("Leyendo datos de calidad...")
    records = parse_quality_from_files(cleaned_dir, unusable_dir)
    print(f"Archivos analizados: {len(records)}")

    usable_count = count_files(cleaned_dir, "txt")
    unusable_count = count_files(unusable_dir, "txt")

    # Generate charts
    sitemap_total = args.sitemap_total
    print(f"Total URLs en sitemap: {sitemap_total}")
    print(f"Fallidos/Repetidos: {sitemap_total - html_count}")
    plot_extracted_files(sitemap_total, html_count, output_dir / "01_archivos_extraidos.png")
    plot_quality_distribution(records, output_dir / "02_distribucion_calidad.png")
    plot_usable_vs_unusable(usable_count, unusable_count, output_dir / "03_usables_vs_no_usables.png")

    print(f"\nEstadisticas generadas en: {output_dir}/")


if __name__ == "__main__":
    main()
