"""
==============================================================
  SMART DAILY ACTIVITY TRACKER WITH DATA ANALYSIS
  & STATISTICAL DISTRIBUTION VISUALIZATION
==============================================================
  Distributions covered:
    1. Normal       - Hours cluster around a daily average
    2. Lognormal    - Skewed, positive-only durations
    3. Gamma        - Total time across multiple tasks
    4. Exponential  - Time between activity entries
    5. Beta         - Hours as proportion of daily budget
==============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

FILE_NAME   = "activities.txt"
DAILY_HOURS = 24.0

PALETTE = {
    "bg":      "#0f0f1a",
    "panel":   "#1a1a2e",
    "accent1": "#e94560",
    "accent2": "#16213e",
    "text":    "#eaeaea",
    "grid":    "#2a2a4a",
    "hist":    "#4cc9f0",
    "normal":  "#4cc9f0",
    "lognorm": "#7209b7",
    "gamma":   "#f72585",
    "expon":   "#4361ee",
    "beta":    "#4ade80",
}

# ── Helpers ───────────────────────────────────────────────────
def hrs_to_display(h: float) -> str:
    """Convert decimal hours → 'Xh Ym' string."""
    total_min = round(h * 60)
    hh, mm = divmod(total_min, 60)
    if hh and mm:
        return f"{hh}h {mm}m"
    elif hh:
        return f"{hh}h"
    else:
        return f"{mm}m"


def prompt_duration() -> float:
    """Ask the user for hours and/or minutes; return total as float hours."""
    print("  Enter time spent (leave blank = 0):")
    try:
        h_str = input("    Hours   : ").strip()
        m_str = input("    Minutes : ").strip()
        hours   = float(h_str) if h_str else 0.0
        minutes = float(m_str) if m_str else 0.0
        if hours < 0 or minutes < 0:
            raise ValueError
        total = hours + minutes / 60.0
        if total <= 0:
            raise ValueError
        return total
    except ValueError:
        return -1.0   # sentinel for bad input

# ── Persistence ───────────────────────────────────────────────
def load_data():
    activities = []
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as fh:
            for line in fh:
                try:
                    name, hours = line.strip().split(",", 1)
                    activities.append((name.strip(), float(hours)))
                except ValueError:
                    pass
    return activities


def save_data(activities):
    with open(FILE_NAME, "w") as fh:
        for name, hours in activities:
            fh.write(f"{name},{hours}\n")

# ── CRUD ──────────────────────────────────────────────────────
def add_activity(activities):
    name = input("  Activity name : ").strip()
    if not name:
        print("  ⚠  Name cannot be empty.\n"); return

    total = prompt_duration()
    if total < 0:
        print("  ⚠  Invalid time — enter non-negative numbers with at least 1 minute total.\n")
        return

    activities.append((name, total))
    save_data(activities)
    print(f"  ✔  '{name}' saved ({hrs_to_display(total)}).\n")


def show_activities(activities):
    if not activities:
        print("  No activities logged yet.\n"); return
    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │              YOUR ACTIVITY LOG                   │")
    print("  ├────┬──────────────────────────┬──────────────────┤")
    print("  │ #  │ Activity                 │ Time             │")
    print("  ├────┼──────────────────────────┼──────────────────┤")
    for i, (name, hours) in enumerate(activities, 1):
        print(f"  │{i:>3} │ {name:<26}│ {hrs_to_display(hours):<16} │")
    print("  └────┴──────────────────────────┴──────────────────┘\n")


def delete_activity(activities):
    show_activities(activities)
    if not activities: return
    try:
        idx = int(input("  Row number to delete (1-based): ")) - 1
        if 0 <= idx < len(activities):
            removed = activities.pop(idx)
            save_data(activities)
            print(f"  ✔  Removed '{removed[0]}'.\n")
        else:
            print("  ⚠  Index out of range.\n")
    except ValueError:
        print("  ⚠  Invalid input.\n")

# ── Analysis ──────────────────────────────────────────────────
def analyze(activities):
    if not activities:
        print("  No data to analyze.\n"); return
    hours   = [h for _, h in activities]
    total   = sum(hours)
    average = np.mean(hours)
    median  = np.median(hours)
    std_dev = np.std(hours, ddof=1) if len(hours) > 1 else 0.0

    print("\n  ╔══════════════════════════════════════════════╗")
    print("  ║           DATA ANALYSIS REPORT               ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Entries       : {len(hours):<26}║")
    print(f"  ║  Total Time    : {hrs_to_display(total):<26}║")
    print(f"  ║  Average Time  : {hrs_to_display(average):<26}║")
    print(f"  ║  Median Time   : {hrs_to_display(median):<26}║")
    print(f"  ║  Std Deviation : {hrs_to_display(std_dev):<26}║")
    print("  ╠══════════════════════════════════════════════╣")

    if average < 1/6:     msg = "Very low — log more tasks!"
    elif average < 0.5:   msg = "Try to be more productive!"
    elif average <= 5:    msg = "Good balance — keep it up!"
    elif average <= 8:    msg = "High output — remember to rest."
    else:                 msg = "Overworked! Take breaks."

    print(f"  ║  Suggestion    : {msg}")
    print("  ╚══════════════════════════════════════════════╝\n")

# ── Plot helpers ──────────────────────────────────────────────
def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PALETTE["panel"])
    ax.set_title(title, color=PALETTE["text"], fontsize=10,
                 fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=PALETTE["text"], fontsize=8)
    ax.set_ylabel(ylabel, color=PALETTE["text"], fontsize=8)
    ax.tick_params(colors=PALETTE["text"], labelsize=7)
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])


def _bins(hours):
    return max(5, len(hours) // 2)


def _overlay_pdf(ax, hours, dist, color, label):
    h = np.array(hours)
    x = np.linspace(max(h.min() * 0.5, 0.001), h.max() * 1.3, 300)
    scale_factor = len(h) * (h.max() - h.min()) / _bins(h)
    try:
        if dist == "norm":
            mu, sigma = stats.norm.fit(h)
            pdf = stats.norm.pdf(x, mu, sigma)
        elif dist == "lognorm":
            s, loc, sc = stats.lognorm.fit(h, floc=0)
            pdf = stats.lognorm.pdf(x, s, loc, sc)
        elif dist == "gamma":
            a, loc, sc = stats.gamma.fit(h, floc=0)
            pdf = stats.gamma.pdf(x, a, loc, sc)
        elif dist == "expon":
            loc, sc = stats.expon.fit(h, floc=0)
            pdf = stats.expon.pdf(x, loc, sc)
        else:
            return
        ax.plot(x, pdf * scale_factor, color=color, linewidth=2.2, label=label)
    except Exception:
        pass


def _overlay_beta(ax, normed, color, label):
    x = np.linspace(0.001, 0.999, 300)
    scale_factor = len(normed) * (normed.max() - normed.min()) / _bins(normed)
    try:
        a, b, loc, sc = stats.beta.fit(normed, floc=0, fscale=1)
        pdf = stats.beta.pdf(x, a, b, loc, sc)
        ax.plot(x, pdf * scale_factor, color=color, linewidth=2.2, label=label)
    except Exception:
        pass


def _fmt_tick(val, _):
    """Format x-axis ticks as 'Xh Ym'."""
    return hrs_to_display(val) if val > 0 else "0"


# ── Main Visualization ────────────────────────────────────────
def show_histogram(activities):
    if not activities:
        print("  No data to display.\n"); return

    hours  = np.array([h for _, h in activities])
    normed = hours / DAILY_HOURS

    fig = plt.figure(figsize=(16, 11), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Smart Daily Activity Tracker — Distribution Analysis",
        color=PALETTE["text"], fontsize=14, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[2, 1.4, 1.4],
        hspace=0.60, wspace=0.38,
        left=0.07, right=0.97,
        top=0.95, bottom=0.06
    )

    # ── Row 0: Master overlay ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(hours, bins=_bins(hours), color=PALETTE["hist"],
             edgecolor="#0f0f1a", alpha=0.75, label="Observed")

    for dist, color, label in [
        ("norm",    PALETTE["normal"],  "Normal"),
        ("lognorm", PALETTE["lognorm"], "Lognormal"),
        ("gamma",   PALETTE["gamma"],   "Gamma"),
        ("expon",   PALETTE["expon"],   "Exponential"),
    ]:
        _overlay_pdf(ax0, hours, dist, color, label)

    scale0 = len(hours) * (hours.max() - hours.min()) / _bins(hours)
    x_b = np.linspace(0.001, 0.999, 300)
    try:
        a, b, loc, sc = stats.beta.fit(normed, floc=0, fscale=1)
        pdf_b = stats.beta.pdf(x_b, a, b, loc, sc)
        ax0.plot(x_b * DAILY_HOURS, pdf_b * scale0,
                 color=PALETTE["beta"], linewidth=2.2, label="Beta (scaled)")
    except Exception:
        pass

    _style_ax(ax0, "All Distributions Overlay — Activity Time",
              "Time Spent", "Frequency")
    ax0.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_tick))
    ax0.legend(fontsize=8, facecolor=PALETTE["panel"],
               labelcolor=PALETTE["text"], framealpha=0.85, loc="upper right")

    # ── Row 1: Normal | Lognormal | Gamma ─────────────────
    for col, (dist, color, title, leg) in enumerate([
        ("norm",    PALETTE["normal"],  "1. Normal Distribution",    "Normal PDF"),
        ("lognorm", PALETTE["lognorm"], "2. Lognormal Distribution", "Lognormal PDF"),
        ("gamma",   PALETTE["gamma"],   "3. Gamma Distribution",     "Gamma PDF"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.hist(hours, bins=_bins(hours), color=PALETTE["hist"],
                edgecolor="#0f0f1a", alpha=0.75, label="Observed")
        _overlay_pdf(ax, hours, dist, color, leg)
        _style_ax(ax, title, "Time Spent", "Frequency")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_tick))
        ax.legend(fontsize=7, facecolor=PALETTE["panel"],
                  labelcolor=PALETTE["text"], framealpha=0.85)

    # ── Row 2: Exponential ────────────────────────────────
    ax_exp = fig.add_subplot(gs[2, 0])
    ax_exp.hist(hours, bins=_bins(hours), color=PALETTE["hist"],
                edgecolor="#0f0f1a", alpha=0.75, label="Observed")
    _overlay_pdf(ax_exp, hours, "expon", PALETTE["expon"], "Exponential PDF")
    _style_ax(ax_exp, "4. Exponential Distribution", "Time Spent", "Frequency")
    ax_exp.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_tick))
    ax_exp.legend(fontsize=7, facecolor=PALETTE["panel"],
                  labelcolor=PALETTE["text"], framealpha=0.85)

    # ── Row 2: Beta ───────────────────────────────────────
    ax_beta = fig.add_subplot(gs[2, 1])
    ax_beta.hist(normed, bins=_bins(normed), color=PALETTE["hist"],
                 edgecolor="#0f0f1a", alpha=0.75, label="Observed (prop.)")
    _overlay_beta(ax_beta, normed, PALETTE["beta"], "Beta PDF")
    _style_ax(ax_beta, "5. Beta Distribution (Proportions)",
              "Fraction of Day (0–1)", "Frequency")
    ax_beta.legend(fontsize=7, facecolor=PALETTE["panel"],
                   labelcolor=PALETTE["text"], framealpha=0.85)

    # ── Row 2: Stats box ──────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.set_facecolor(PALETTE["panel"])
    ax_stats.axis("off")

    std = hours.std(ddof=1) if len(hours) > 1 else 0.0
    rows = [
        ("QUICK STATS", ""),
        ("─" * 20, ""),
        ("Count",   str(len(hours))),
        ("Min",     hrs_to_display(hours.min())),
        ("Max",     hrs_to_display(hours.max())),
        ("Mean",    hrs_to_display(hours.mean())),
        ("Median",  hrs_to_display(float(np.median(hours)))),
        ("Std Dev", hrs_to_display(std)),
        ("Total",   hrs_to_display(hours.sum())),
    ]
    text_block = "\n".join(
        f"  {k:<10} {v}" if v else f"  {k}" for k, v in rows
    )
    ax_stats.text(
        0.08, 0.92, text_block,
        transform=ax_stats.transAxes,
        color=PALETTE["text"], fontsize=9, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor=PALETTE["accent2"],
                  edgecolor=PALETTE["accent1"],
                  linewidth=1.5)
    )

    plt.savefig("activity_histogram.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.show()
    print("  Chart saved as 'activity_histogram.png'.\n")

# ── Demo data ─────────────────────────────────────────────────
def load_demo_data(activities):
    demo = [
        ("Morning Exercise",    1 + 15/60),   # 1h 15m
        ("Deep Work - Coding",  4 + 30/60),   # 4h 30m
        ("Lunch Break",         45/60),        # 45m
        ("Workout",             1.0),
        ("Family Time",         3 + 30/60),
    ]
    for item in demo:
        activities.append(item)
    save_data(activities)
    print(f"  ✔  {len(demo)} demo activities loaded.\n")

# ── CLI ───────────────────────────────────────────────────────
MENU = """
  ╔══════════════════════════════════════════╗
  ║    SMART DAILY ACTIVITY TRACKER          ║
  ╠══════════════════════════════════════════╣
  ║  1. Add Activity                         ║
  ║  2. Show Activities                      ║
  ║  3. Analyze Data                         ║
  ║  4. Show Histogram (all distributions)   ║
  ║  5. Delete an Activity                   ║
  ║  6. Load Demo Data                       ║
  ║  7. Exit                                 ║
  ╚══════════════════════════════════════════╝
  Choose: """


def main():
    activities = load_data()
    print("\n  Welcome to the Smart Daily Activity Tracker!")
    if not activities:
        print("  Tip: Choose 6 to load demo data, then 4 to see the charts.\n")

    while True:
        choice = input(MENU).strip()
        if   choice == "1": add_activity(activities)
        elif choice == "2": show_activities(activities)
        elif choice == "3": analyze(activities)
        elif choice == "4": show_histogram(activities)
        elif choice == "5": delete_activity(activities)
        elif choice == "6": load_demo_data(activities)
        elif choice == "7":
            print("\n  Goodbye! Stay productive.\n"); break
        else:
            print("  ⚠  Invalid choice. Enter 1–7.\n")


if __name__ == "__main__":
    main()
