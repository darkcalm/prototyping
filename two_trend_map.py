import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl

# Axes definition
axes = {
    "x_left": "Centralized facility",
    "x_right": "Decentralized point-of-disposal",
    "y_bottom": "Material signature sensing",
    "y_top": "Symbolic identifier capture",
}

# Points as before
points = [
    ("TOMRA AUTOSORT", -0.92, -0.92, "Existing"),
    ("STEINERT KSS/XRF", -0.90, -0.90, "Existing"),
    ("Pellenc ST Mistral+", -0.85, -0.85, "Existing"),
    ("MSS FiberMax/Vivid AI", -0.82, -0.84, "Existing"),
    ("Specim HSI (FX17/FX50)", -0.80, -0.80, "Existing"),
    ("Bollegraaf RoBB QC", -0.82, -0.82, "Existing"),
    ("STEINERT ISS (Inductive)", -0.83, -0.78, "Existing"),
    ("STEINERT FinesMaster (Eddy)", -0.80, -0.76, "Existing"),
    ("Air Classification", -0.78, -0.72, "Existing"),
    ("AMP Robotics", -0.70, -0.70, "Existing"),
    ("Waste Robotics SamurAI", -0.70, -0.70, "Existing"),
    ("ZenRobotics ZRR", -0.70, -0.68, "Existing"),
    ("Stadler Stockholm Plant", -0.92, -0.75, "Existing"),
    ("Machinex Turnkey", -0.90, -0.82, "Existing"),
    ("CP Group Turnkey", -0.95, -0.86, "Existing"),
    ("GreyParrot Analyzer", -0.58, -0.20, "Existing"),
    ("Recycleye Vision", -0.60,  0.20, "Existing"),
    ("TrueCircle Analytics", -0.50, 0.00, "Existing"),
    ("Lixo Analytics", -0.45, 0.05, "Existing"),
    ("Envac Optical Sorting (bags)", -0.40, 0.70, "Existing"),
    ("CleanRobotics TrashBot", 0.72, -0.50, "Existing"),
    ("Ganiga Hoooly", 0.82, -0.40, "Existing"),
    ("Pello by RTS", 0.60, -0.20, "Existing"),
    ("Smart Waste Bins (IoT)", 0.55, -0.10, "Existing"),
    ("Sensoneo DRS/Smart Waste", 0.55, 0.65, "Existing"),
    ("WasteHero Platform", 0.40, 0.20, "Existing"),
    ("Stadler Sollenau PreZero Plant", -0.85, -0.78, "Existing"),
    ("Stadler Global Integration", -0.88, -0.80, "Existing"),
    ("EverestLabs MRF Suite", -0.68, -0.55, "Existing"),
    ("Recycleye Robotics (line pick)", -0.66, -0.50, "Existing"),
    ("HSI + AI (RECLAIM)", -0.65, -0.40, "Research"),
    ("RL Multi-Robot Sorting", -0.62, -0.45, "Research"),
    ("Enzymatic PET Degradation", -0.40, -0.90, "Visionary"),
    ("Barcode-Scan Bins (Moonshot)", 0.95, 0.94, "Moonshot"),
]

# Two trend lines - approximated as single straight lines
# Trend 1 (centralized + material signature up to Recycleye Vision)
x1 = [-0.95, -0.60]
y1 = [-0.90, 0.20]

# Trend 2 (decentralized + symbolic) shorter segment same direction
x2 = [-0.35, 0.75]  # shortened from 0.95 to 0.75 on X axis
y2 = [-0.65, 0.72]  # shortened from 0.94 to 0.72 on Y axis

# Regions as before
no_go_center = (-0.85, 0.85)
no_go_rx, no_go_ry = 0.12, 0.10

white_spot_center = (0.86, 0.88)
white_spot_rx, white_spot_ry = 0.12, 0.10

colors = {
    "Existing": "#4aa3df",
    "Research": "#f6c343",
    "Visionary": "#bc8cff",
    "Moonshot": "#ffd166",
}

mpl.rcParams["figure.figsize"] = (10, 9)
fig, ax = plt.subplots()
ax.set_facecolor("#17191c")
fig.patch.set_facecolor("#17191c")

# Draw regions
no_go = Ellipse(xy=no_go_center, width=2*no_go_rx, height=2*no_go_ry,
                edgecolor="red", linestyle=(0, (6, 6)), facecolor="none", linewidth=2, zorder=1)
ax.add_patch(no_go)
white_spot = Ellipse(xy=white_spot_center, width=2*white_spot_rx, height=2*white_spot_ry,
                     edgecolor="white", linestyle=(0, (6, 6)), facecolor="none", linewidth=2, zorder=1)
ax.add_patch(white_spot)

# Plot two trend lines
ax.plot(x1, y1, color="#9aa0a6", linewidth=2, zorder=2, label='Trend 1: Centralized+Material')
ax.plot(x2, y2, color="#d6a500", linewidth=2, zorder=2, label='Trend 2: Decentralized+Symbolic')
ax.scatter(x1+y2, y1+y2, color="#9aa0a6", s=20, zorder=3)
ax.scatter(x2+y2, y2+y2, color="#d6a500", s=20, zorder=3)

# Plot points
for name, x, y, cat in points:
    if cat == "Moonshot":
        ax.scatter([x], [y], s=160, marker="*", color=colors[cat], edgecolor="black", linewidth=0.8, zorder=5)
    else:
        ax.scatter([x], [y], s=60, marker="o", color=colors.get(cat, "#cccccc"),
                   edgecolor="black", linewidth=0.6, zorder=4)
    ax.text(x + 0.015, y + 0.015, name, fontsize=8, color="white", zorder=6)

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.axhline(0, color="#2a2e33", linewidth=1)
ax.axvline(0, color="#2a2e33", linewidth=1)
ax.set_xlabel(f"{axes['x_left']}  ←  X  →  {axes['x_right']}", color="white")
ax.set_ylabel(f"{axes['y_bottom']}  ↑  Y  ↓  {axes['y_top']}", color="white")

ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.tick_params(colors="white")

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="w", label="Existing", markerfacecolor=colors["Existing"], markeredgecolor="black", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Research", markerfacecolor=colors["Research"], markeredgecolor="black", markersize=8),
    Line2D([0], [0], marker="o", color="w", label="Visionary", markerfacecolor=colors["Visionary"], markeredgecolor="black", markersize=8),
    Line2D([0], [0], marker="*", color="w", label="Moonshot", markerfacecolor=colors["Moonshot"], markeredgecolor="black", markersize=12),
    Line2D([0], [0], color="red", linestyle=(0, (6, 6)), label="No-go (dashed red)"),
    Line2D([0], [0], color="white", linestyle=(0, (6, 6)), label="White spot (dashed white)"),
    Line2D([0], [0], color="#9aa0a6", linewidth=2, label="Trend 1"),
    Line2D([0], [0], color="#d6a500", linewidth=2, label="Trend 2"),
]
leg = ax.legend(handles=legend_elems, facecolor="#22262b", edgecolor="#42464b", labelcolor="white", loc="lower right")
for text in leg.get_texts():
    text.set_color("white")

plt.tight_layout()
plt.savefig("two_trend_map.png", dpi=200)
plt.show()
