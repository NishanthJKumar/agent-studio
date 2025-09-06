import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")

# Data
# Test task success rate
x1 = [0, 232, 472]
y1 = [0, 75, 75]
x2 = [0, 232, 472]
y2 = [0, 25, 50]

# Train task success rate
x3 = [0, 232, 472]
y3 = [50, 75, 87.5]
x4 = [0, 232, 472]
y4 = [25, 50, 75]

# Figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot test lines
ax.plot(
    x1, y1, marker="o", markersize=8, linewidth=2.5,
    color="#1f77b4", label="scoregen (ours)"
)
ax.plot(
    x2, y2, marker="s", markersize=8, linewidth=2.5,
    color="#ff7f0e", label="naivegen"
)
# Baseline reference
ax.axhline(y=50, color="gray", linestyle="--", linewidth=2,
           label="direct prompting (baseline)")

# Axis labels
ax.set_xlabel("# datapoints", fontsize=14)
ax.set_ylabel("Test task success rate", fontsize=14)

# Lock y-axis from 0% to 100%
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

# Tick settings
ax.tick_params(axis="both", which="major", labelsize=12)

# Legend
ax.legend(fontsize=12, frameon=False)

# Tight layout
fig.tight_layout()

plt.show()


# Plot train lines
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    x3, y3, marker="o", markersize=8, linewidth=2.5,
    color="#1f77b4", label="scoregen (ours)"
)
ax.plot(
    x4, y4, marker="s", markersize=8, linewidth=2.5,
    color="#ff7f0e", label="naivegen"
)
# Baseline reference
ax.axhline(y=25, color="gray", linestyle="--", linewidth=2,
           label="direct prompting (baseline)")
# Axis labels
ax.set_xlabel("# datapoints", fontsize=14)
ax.set_ylabel("Test task success rate", fontsize=14)

# Lock y-axis from 0% to 100%
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

# Tick settings
ax.tick_params(axis="both", which="major", labelsize=12)

# Legend
ax.legend(fontsize=12, frameon=False)

# Tight layout
fig.tight_layout()

plt.show()
