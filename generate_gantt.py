import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

tasks = [
    {"name": "Kickoff & Topic Confirmation", "start": "2025-09-01", "end": "2025-09-30", "status": "Completed"},
    {"name": "Dataset Selection & Download", "start": "2025-09-05", "end": "2025-10-10", "status": "Completed"},
    {"name": "EDA & Data Cleaning", "start": "2025-10-01", "end": "2025-10-20", "status": "Completed"},
    {"name": "Baseline Models: RF / XGB / LGBM / CB", "start": "2025-10-10", "end": "2025-10-30", "status": "Completed"},
    {"name": "Encoding & Imbalance Strategy", "start": "2025-10-20", "end": "2025-11-10", "status": "Completed"},
    {"name": "Proposal Writing & Submission (DDL 11/01)", "start": "2025-10-20", "end": "2025-11-01", "status": "Completed"},
    {"name": "Tuning & Early Stopping – Round 1", "start": "2025-11-05", "end": "2025-11-25", "status": "Completed"},
    {"name": "Main Results & Stability Check", "start": "2025-11-20", "end": "2025-12-05", "status": "Completed"},
    {"name": "Progress Report (DDL 12/01)", "start": "2025-11-25", "end": "2025-12-01", "status": "Completed"},
    {"name": "Mid-term Presentation Prep", "start": "2025-12-02", "end": "2025-12-12", "status": "Completed"},
    {"name": "January: Review / Refactor / Pipeline Hardening", "start": "2025-12-13", "end": "2026-02-10", "status": "In Progress"},
    {"name": "Semester 2: Deep Experiments & System Polish", "start": "2026-02-24", "end": "2026-03-31", "status": "Pending"},
    {"name": "Explainability & Ablations", "start": "2026-03-01", "end": "2026-03-25", "status": "Pending"},
    {"name": "Final Thesis Writing (Rolling → Final, DDL 04/12)", "start": "2026-03-10", "end": "2026-04-12", "status": "Pending"},
    {"name": "Defense Deck & Demo Preparation", "start": "2026-04-10", "end": "2026-04-20", "status": "Pending"},
    {"name": "Oral Defense", "start": "2026-04-20", "end": "2026-04-24", "status": "Pending"},
    {"name": "Revisions & Archival Submission (DDL 04/27)", "start": "2026-04-24", "end": "2026-04-27", "status": "Pending"},
]

df = pd.DataFrame(tasks)
df["start_dt"] = pd.to_datetime(df["start"])
df["end_dt"] = pd.to_datetime(df["end"])

colors = {"Completed": "#2ecc71", "In Progress": "#f1c40f", "Pending": "#95a5a6"}

os.makedirs(os.path.join("outputs", "gantt"), exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 8))
ypos = list(range(len(df)))[::-1]
start_nums = mdates.date2num(df["start_dt"]) if hasattr(df["start_dt"], "__array__") else mdates.date2num(pd.to_datetime(df["start_dt"]))
end_nums = mdates.date2num(df["end_dt"]) if hasattr(df["end_dt"], "__array__") else mdates.date2num(pd.to_datetime(df["end_dt"]))
widths = end_nums - start_nums

span = max(end_nums) - min(start_nums)
x_label = max(end_nums) + span * 0.08
for i, y in enumerate(ypos):
    status = df["status"].iloc[i]
    ax.barh(y, widths[i], left=start_nums[i], color=colors.get(status, "#95a5a6"), edgecolor="black")
    ax.text(x_label, y, f"{df['name'].iloc[i]} — {status}", va="center", ha="left", fontsize=9)

ax.set_yticks([])
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=30)
plt.grid(axis="x", linestyle="--", alpha=0.3)

today = dt.date.today()
today_num = mdates.date2num(pd.to_datetime(today))
ax.axvline(today_num, color="#e74c3c", linestyle="--", linewidth=2)
ax.text(today_num, len(df), f"Today: {today}", color="#e74c3c", ha="right", va="bottom")

handles = [plt.Rectangle((0,0),1,1,color=colors[s]) for s in colors]
labels = list(colors.keys())
ax.legend(handles, labels, loc="upper left")

ax.set_xlim(min(start_nums), max(end_nums) + span * 0.25)
plt.tight_layout()
plt.savefig(os.path.join("outputs", "gantt", "gantt.png"), dpi=200)
plt.savefig(os.path.join("outputs", "gantt", "gantt.pdf"))
pd.DataFrame(tasks).to_csv(os.path.join("outputs", "gantt", "gantt_tasks.csv"), index=False)
print("Saved to outputs/gantt/")
