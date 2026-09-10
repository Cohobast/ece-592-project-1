import pandas as pd
import matplotlib.pyplot as plt

MACHINE_NAME = "Crux"
CACHE_LEVEL = "l1"

FILE_NAME = "crux_assoc_assoc_summary.csv"

df = pd.read_csv(FILE_NAME)

plt.plot(
    df["ways"],
    df["median_tsc_ticks_per_access"],
    linewidth=1.5
)

ax = plt.gca()

plt.xlabel(
    "Ways",
    fontsize=11
)

plt.ylabel(
    "Median Latency (TSC ticks/access)",
    fontsize=11
)

plt.title(
    MACHINE_NAME + ": Pointer-Chase Latency vs. Ways",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    MACHINE_NAME.lower()+"_"+CACHE_LEVEL.lower()+"_assoc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    MACHINE_NAME.lower()+"_"+CACHE_LEVEL.lower()+"_assoc.pdf",
    bbox_inches="tight"
)

plt.show()

