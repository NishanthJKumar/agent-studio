import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Or 'Qt5Agg', 'MacOSX', etc.

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")

# Data
x1 = [0, 232, 472]
y1 = [0, 75, 75]
x2 = [0, 232, 472]
y2 = [0, 25, 50]

# Figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot lines
ax.plot(
    x1, y1, marker="o", markersize=8, linewidth=2.5,
    color="#1f77b4", label="scoregen (ours)"
)
ax.plot(
    x2, y2, marker="s", markersize=8, linewidth=2.5,
    color="#ff7f0e", label="naivegen"
)

# Axis labels
ax.set_xlabel("# datapoints", fontsize=14)
ax.set_ylabel("Test task success rate", fontsize=14)

# Tick settings
ax.tick_params(axis="both", which="major", labelsize=12)

# Legend
ax.legend(fontsize=12, frameon=False)

# Tight layout
fig.tight_layout()

plt.show()
