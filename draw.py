import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Data
# -----------------------------
data = [
    ["BPIC2017_W","CFLD",0.47,0.43,0.60,0.63,0.36,0.28],
    ["BPIC2017_W","3GD",0.69,0.67,0.86,0.87,0.58,0.39],
    ["BPIC2017_W","AED",6638.97,7260.51,1853.17,50375.91,856.43,240.26],
    ["BPIC2017_W","CED",1.74,2.29,3.00,4.85,1.73,1.69],
    ["BPIC2017_W","RED",622.49,985.51,1093.79,980.66,619.76,207.98],
    ["BPIC2017_W","CAR",4946.49,4655.27,595.23,37177.30,19.46,124.77],
    ["BPIC2017_W","CTD",266.85,173.18,231.42,222.06,114.31,77.70],

    ["BPIC2012_W","CFLD",0.47,0.30,0.41,0.67,0.49,0.42],
    ["BPIC2012_W","3GD",0.56,0.37,0.50,0.82,0.65,0.48],
    ["BPIC2012_W","AED",535.81,255.79,577.17,78667.23,435.40,666.49],
    ["BPIC2012_W","CED",2.40,2.42,5.73,5.23,2.31,2.22],
    ["BPIC2012_W","RED",488.65,102.71,390.73,169.39,390.24,391.41],
    ["BPIC2012_W","CAR",84.14,181.81,19067.00,78786.42,39.52,119.48],
    ["BPIC2012_W","CTD",186.93,88.70,193.02,182.80,148.94,117.43],

    ["CVS","CFLD",0.24,0.34,0.43,0.83,0.30,0.24],
    ["CVS","3GD",0.40,0.70,0.83,0.98,0.47,0.43],
    ["CVS","AED",4106.01,1552.40,734.37,15124.88,46.62,106.20],
    ["CVS","CED",1.68,1.74,2.26,5.33,1.85,2.39],
    ["CVS","RED",215.78,205.01,223.89,162.90,30.44,66.49],
    ["CVS","CAR",3952.53,1306.99,676.48,15261.32,20.37,25.69],
    ["CVS","CTD",297.46,183.16,234.21,260.90,52.42,33.05],

    ["ACR","CFLD",0.35,0.21,0.22,0.89,0.18,0.22],
    ["ACR","3GD",0.55,0.27,0.34,0.96,0.25,0.46],
    ["ACR","AED",1051.14,851.19,607.67,41244.01,562.57,75.83],
    ["ACR","CED",2.10,7.22,6.53,4.95,2.15,2.00],
    ["ACR","RED",216.57,215.28,50.55,106.79,220.34,16.63],
    ["ACR","CAR",572.52,599.50,597.40,43016.72,260.68,66.62],
    ["ACR","CTD",63.30,84.42,82.76,64.84,48.67,46.87],

    ["P2P","CFLD",0.58,0.53,0.65,0.86,0.22,0.32],
    ["P2P","3GD",0.50,0.73,0.66,1.00,0.24,0.33],
    ["P2P","AED",6683.99,1530.53,6823.04,193809.23,1449.10,2088.00],
    ["P2P","CED",6.73,2.84,7.46,5.32,0.62,1.47],
    ["P2P","RED",4175.01,880.21,5510.40,4034.39,847.74,1431.92],
    ["P2P","CAR",1014.59,1113.68,971.85,86179.88,929.00,883.73],
    ["P2P","CTD",559.34,666.93,637.72,612.81,555.29,553.68],

    ["MP","CFLD",0.76,0.77,0.74,0.91,0.76,0.69],
    ["MP","3GD",0.87,0.83,0.79,0.98,0.86,0.78],
    ["MP","AED",2180.93,270.76,969.03,131458.03,350.36,629.88],
    ["MP","CED",2.64,3.62,8.82,4.36,2.09,3.22],
    ["MP","RED",231.41,164.79,692.78,317.60,199.64,360.98],
    ["MP","CAR",1340.11,196.42,196.42,78032.07,165.32,27.22],
    ["MP","CTD",90.10,98.09,88.22,44.04,34.11,31.96],
]
methods = ["SIMOD","LSTM","GRU","LSTM(GAN)","DSIM","DPSIM"]
logs = ["BPIC2017_W","BPIC2012_W","CVS","ACR","P2P","MP"]
df = pd.DataFrame(data, columns=["log", "metric"] + methods)

groups = {
    "control_flow": ["CFLD", "3GD"],
    "temporal": ["AED", "CED", "RED"],
    "congestion": ["CAR", "CTD"],
}
titles = {
    "control_flow": "Panel (a) Control-flow perspective",
    "temporal": "Panel (b) Temporal perspective",
    "congestion": "Panel (c) Congestion perspective",
}

outdir = Path("/mnt/data/panel_compare_figs")
outdir.mkdir(exist_ok=False)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Helpers
# -----------------------------
def format_val(v):
    if abs(v) >= 1000:
        return f"{v:.0f}"
    elif abs(v) >= 100:
        return f"{v:.1f}"
    else:
        return f"{v:.2f}"

def get_font(size=28):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def combine_images_vertically(image_paths, title, output_path, bg="white", pad=24):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    font_title = get_font(30)
    font_sub = get_font(22)

    canvas_w = max(widths) + 2 * pad
    title_h = 64
    total_h = title_h + sum(heights) + pad * (len(images) + 1)

    canvas = Image.new("RGB", (canvas_w, total_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill="black", font=font_title)

    y = title_h + pad
    for idx, img in enumerate(images, start=1):
        x = (canvas_w - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + pad

    canvas.save(output_path)

def combine_three_panels_horizontal(panel_paths, output_path, bg="white", pad=24):
    images = [Image.open(p).convert("RGB") for p in panel_paths]
    font_title = get_font(34)
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    title_h = 70
    canvas_w = sum(widths) + pad * (len(images) + 1)
    canvas_h = max(heights) + title_h + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "Three-panel comparison", fill="black", font=font_title)
    x = pad
    for im in images:
        y = title_h + (max(heights) - im.height) // 2 + pad
        canvas.paste(im, (x, y))
        x += im.width + pad
    canvas.save(output_path)

# -----------------------------
# 1) Dumbbell plots: DSIM vs DPSIM
# -----------------------------
def make_dumbbell_plot(metric, save_path):
    sub = df[df["metric"] == metric].set_index("log").loc[logs]
    dsim = sub["DSIM"].values
    dpsim = sub["DPSIM"].values
    y = np.arange(len(logs))

    fig = plt.figure(figsize=(8.5, 3.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)

    for i, (x1, x2) in enumerate(zip(dsim, dpsim)):
        ax.plot([x1, x2], [i, i], linewidth=2)
    ax.plot(dsim, y, "o", markersize=7, label="DSIM")
    ax.plot(dpsim, y, "o", markersize=7, label="DPSIM")

    ax.set_yticks(y)
    ax.set_yticklabels(logs)
    ax.invert_yaxis()
    ax.set_xlabel("Original value (smaller is better)")
    ax.set_title(metric)

    # annotate values
    xmax = max(dsim.max(), dpsim.max())
    xmin = min(dsim.min(), dpsim.min())
    span = xmax - xmin if xmax > xmin else 1
    offset = span * 0.02
    for i, (x1, x2) in enumerate(zip(dsim, dpsim)):
        ax.text(x1 + offset, i - 0.12, format_val(x1), fontsize=8)
        ax.text(x2 + offset, i + 0.18, format_val(x2), fontsize=8)

    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 2) Grouped dot plots: all methods
# -----------------------------
def make_grouped_dot_plot(metric, save_path):
    sub = df[df["metric"] == metric].set_index("log").loc[logs, methods]
    y_base = np.arange(len(logs))
    offsets = np.linspace(-0.25, 0.25, len(methods))

    fig = plt.figure(figsize=(8.8, 4.2))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)

    # draw methods in sequence without manual colors
    for idx, method in enumerate(methods):
        vals = sub[method].values
        y = y_base + offsets[idx]
        size = 70 if method in ["DSIM", "DPSIM"] else 35
        ax.scatter(vals, y, s=size, label=method)

    ax.set_yticks(y_base)
    ax.set_yticklabels(logs)
    ax.invert_yaxis()
    ax.set_xlabel("Original value (smaller is better)")
    ax.set_title(metric)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Generate charts and combine panels
# -----------------------------
dumbbell_panel_paths = []
groupdot_panel_paths = []

for group_name, metric_list in groups.items():
    # dumbbell
    dumbbell_metric_paths = []
    for metric in metric_list:
        p = outdir / f"dumbbell_{group_name}_{metric}.png"
        make_dumbbell_plot(metric, p)
        dumbbell_metric_paths.append(p)
    panel_path = outdir / f"dumbbell_panel_{group_name}.png"
    combine_images_vertically(dumbbell_metric_paths, titles[group_name], panel_path)
    dumbbell_panel_paths.append(panel_path)

    # grouped dot
    groupdot_metric_paths = []
    for metric in metric_list:
        p = outdir / f"groupdot_{group_name}_{metric}.png"
        make_grouped_dot_plot(metric, p)
        groupdot_metric_paths.append(p)
    panel_path2 = outdir / f"groupdot_panel_{group_name}.png"
    combine_images_vertically(groupdot_metric_paths, titles[group_name], panel_path2)
    groupdot_panel_paths.append(panel_path2)

# final combined figures
final_dumbbell = outdir / "three_panel_dumbbell.png"
final_groupdot = outdir / "three_panel_grouped_dot.png"
combine_three_panels_horizontal(dumbbell_panel_paths, final_dumbbell)
combine_three_panels_horizontal(groupdot_panel_paths, final_groupdot)

print("Generated files:")
for p in [final_dumbbell, final_groupdot] + dumbbell_panel_paths + groupdot_panel_paths:
    print(str(p))
