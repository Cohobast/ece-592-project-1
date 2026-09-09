import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 0. CONFIGURATION
#
# Point these at the *_stride_summary.csv files produced by
# cache_line_assoc_x86.c (mode "stride"). Each summary file can
# contain multiple candidate strides at once (from a comma-separated
# STRIDE_LIST), so a single coarse run and one or more dense
# zoom-in runs are enough.
#
#   coarse run: cache_line_assoc_x86 stride sweep "8,16,32,64,128,256,512" \
#               1024 131072 1000000 128 12345 data/line_coarse
#
#   dense run(s), zoomed around a suspected knee, e.g.:
#               cache_line_assoc_x86 stride sweep "16,24,32,40,48,56,64,80,96,128" \
#               4096 65536 1000000 128 12345 data/line_dense_knee1
# ============================================================

COARSE_FILE = "line_size_stride_summary.csv"
DENSE_GLOB = "line_dense_*_stride_summary.csv"

MACHINE_NAME = "Crux"


# ============================================================
# 1. READ THE COARSE STRIDE SWEEP
# ============================================================

coarse = pd.read_csv(COARSE_FILE)

# Sort by stride, then by working-set size, so each stride's curve
# connects its points in increasing footprint order.
coarse = coarse.sort_values(["stride_bytes", "actual_footprint_bytes"])


# ============================================================
# 2. READ ALL DENSE STRIDE MEASUREMENTS
# ============================================================

dense_files = glob.glob(DENSE_GLOB)

dense_data = []

for file in dense_files:
    df = pd.read_csv(file)
    dense_data.append(df)

if dense_data:
    dense = pd.concat(dense_data, ignore_index=True)
else:
    dense = pd.DataFrame(columns=coarse.columns)

dense = dense.sort_values(["stride_bytes", "actual_footprint_bytes"])


# ============================================================
# 3. COMBINE COARSE + DENSE DATA FOR THE CONNECTING LINES
# ============================================================

all_data = pd.concat(
    [coarse, dense],
    ignore_index=True
)

all_data = all_data.sort_values(["stride_bytes", "actual_footprint_bytes"])

strides = sorted(all_data["stride_bytes"].unique())


# ============================================================
# 4. PRINT FILES AND DATA FOR VERIFICATION
# ============================================================

print("Coarse file:")
print(" ", COARSE_FILE)

print("\nDense files:")
for file in sorted(dense_files):
    print(" ", file)

print("\nStrides found (bytes):", strides)

print("\nData being plotted:")

print(
    all_data[
        [
            "stride_bytes",
            "actual_footprint_bytes",
            "median_tsc_ticks_per_access",
            "mean_tsc_ticks_per_access"
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. FORMAT BYTES AS KiB / MiB
# ============================================================

def format_bytes(x, pos):

    if x >= 1024**2:
        return f"{x / 1024**2:g} MiB"

    elif x >= 1024:
        return f"{x / 1024:g} KiB"

    else:
        return f"{x:g} B"


def format_stride_label(stride_bytes):
    if stride_bytes >= 1024:
        return f"{stride_bytes / 1024:g} KiB stride"
    return f"{stride_bytes:g} B stride"


# ============================================================
# 6. CREATE FIGURE
# ============================================================

plt.figure(figsize=(11, 6.5))
ax = plt.gca()


# ============================================================
# 7. DISTINGUISHABLE MARKERS/LINE STYLES PER STRIDE
#
# The homework's required figure style is conference/old-ISCA style:
# grayscale-safe, clearly distinguishable marker shapes and line
# styles rather than relying on color alone, no background grid.
# ============================================================

markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
linestyles = ["-", "--", "-.", ":"]
grayscale = [
    str(shade) for shade in
    [0.0, 0.55, 0.15, 0.7, 0.3, 0.4, 0.05, 0.6, 0.2, 0.5]
]


# ============================================================
# 8. PLOT ONE FAMILY-OF-CURVES LINE PER CANDIDATE STRIDE
#
# If a dense re-run re-measures a footprint that the coarse sweep
# already covered, the two independent measurements will not be
# identical. Drawing a single connecting line straight through both
# (in whatever row order they happen to sort to) produces a
# misleading zig-zag/spike at that footprint. To avoid that, the
# connecting line is drawn through one point per unique footprint
# (the mean of any repeated measurements at that footprint), while
# every individual coarse and dense measurement is still shown as
# its own marker so no data is hidden.
# ============================================================

for i, stride_bytes in enumerate(strides):
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    color = grayscale[i % len(grayscale)]
    label = format_stride_label(stride_bytes)

    stride_all = all_data[all_data["stride_bytes"] == stride_bytes]
    stride_coarse = coarse[coarse["stride_bytes"] == stride_bytes]
    stride_dense = dense[dense["stride_bytes"] == stride_bytes]

    stride_line = (
        stride_all
        .groupby("actual_footprint_bytes", as_index=False)["median_tsc_ticks_per_access"]
        .mean()
        .sort_values("actual_footprint_bytes")
    )

    # Trend line for this stride (no markers -- markers are drawn
    # separately below so coarse vs. dense stays visually distinct).
    plt.plot(
        stride_line["actual_footprint_bytes"],
        stride_line["median_tsc_ticks_per_access"],
        linewidth=1.5,
        linestyle=linestyle,
        color=color,
        zorder=3
    )

    # Coarse measurements: filled marker. This scatter call also
    # carries the legend entry for this stride.
    plt.scatter(
        stride_coarse["actual_footprint_bytes"],
        stride_coarse["median_tsc_ticks_per_access"],
        marker=marker,
        s=45,
        color=color,
        label=label,
        zorder=4
    )

    # Dense re-measurements: open/outlined marker overlay so they
    # remain visually identifiable as the denser, zoomed-in samples
    # without needing a second legend entry per stride.
    if not stride_dense.empty:
        plt.scatter(
            stride_dense["actual_footprint_bytes"],
            stride_dense["median_tsc_ticks_per_access"],
            marker=marker,
            s=70,
            facecolors="none",
            edgecolors=color,
            linewidths=1.3,
            zorder=5
        )


# ============================================================
# 9. USE LOG BASE-2 X-AXIS
# ============================================================

plt.xscale("log", base=2)

ax.xaxis.set_major_formatter(
    FuncFormatter(format_bytes)
)


# ============================================================
# 10. NO BACKGROUND GRID LINES (required figure style)
# ============================================================

ax.grid(False)


# ============================================================
# 11. AXIS LABELS AND TITLE
# ============================================================

plt.xlabel(
    "Working-Set Size (Footprint)",
    fontsize=11
)

plt.ylabel(
    "Median Latency (TSC ticks/access)",
    fontsize=11
)

plt.title(
    f"{MACHINE_NAME}: Pointer-Chase Latency vs. Working-Set Size, "
    f"by Candidate Stride",
    fontsize=13
)


# ============================================================
# 12. LEGEND
#
# One entry per candidate stride (filled marker = coarse/connecting
# line, open marker overlay in the plot = dense re-measurement).
# ============================================================

plt.legend(
    loc="upper left",
    frameon=True,
    title="Candidate byte jump"
)


# ============================================================
# 13. CLEAN UP FIGURE SPACING
# ============================================================

plt.tight_layout()


# ============================================================
# 14. SAVE HIGH-QUALITY PNG
# ============================================================

plt.savefig(
    f"{MACHINE_NAME.lower()}_stride_line_size.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. SAVE VECTOR PDF FOR REPORT
# ============================================================

plt.savefig(
    f"{MACHINE_NAME.lower()}_stride_line_size.pdf",
    bbox_inches="tight"
)


# ============================================================
# 16. DISPLAY GRAPH
# ============================================================

plt.show()