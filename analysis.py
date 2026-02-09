import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "skygeni_sales_data.csv"
OUTDIR = Path("skygeni_charts")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["created_date"] = pd.to_datetime(df["created_date"])
df["closed_date"] = pd.to_datetime(df["closed_date"])
df["won"] = (df["outcome"].str.lower() == "won").astype(int)
df["close_month"] = df["closed_date"].dt.to_period("M").dt.to_timestamp()
df["close_quarter"] = df["closed_date"].dt.to_period("Q").astype(str)

# Monthly trends
m = df.groupby("close_month").agg(
    deals=("deal_id", "count"),
    win_rate=("won", "mean"),
    avg_cycle=("sales_cycle_days", "mean"),
).sort_index()

plt.figure(figsize=(9, 4.8))
plt.plot(m.index, m["win_rate"], marker="o")
plt.ylabel("Win Rate")
plt.xlabel("Close Month")
plt.ylim(0, 1)
plt.title("Win Rate Trend (Monthly)")
plt.tight_layout()
plt.savefig(OUTDIR / "win_rate_trend_monthly.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 4.8))
plt.plot(m.index, m["avg_cycle"], marker="o")
plt.ylabel("Avg Sales Cycle (days)")
plt.xlabel("Close Month")
plt.title("Sales Cycle Trend (Monthly)")
plt.tight_layout()
plt.savefig(OUTDIR / "sales_cycle_trend_monthly.png", dpi=200)
plt.close()

# Lead source Q1 vs Q2 2024
focus = df[df["close_quarter"].isin(["2024Q1", "2024Q2"])].copy()
lead = focus.groupby(["close_quarter", "lead_source"]).agg(
    win_rate=("won", "mean")
).reset_index()

pivot = lead.pivot(index="lead_source", columns="close_quarter", values="win_rate").reindex(
    ["Inbound", "Outbound", "Partner", "Referral"]
)

plt.figure(figsize=(8.5, 4.8))
x = np.arange(len(pivot.index))
width = 0.38
plt.bar(x - width / 2, pivot["2024Q1"], width, label="2024Q1")
plt.bar(x + width / 2, pivot["2024Q2"], width, label="2024Q2")
plt.xticks(x, pivot.index)
plt.ylabel("Win Rate")
plt.ylim(0, 1)
plt.title("Win Rate by Lead Source (2024Q1 vs 2024Q2)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "lead_source_win_rate_q1_q2.png", dpi=200)
plt.close()

# Stage Stall Index Heatmap
overall = focus.groupby("close_quarter")["sales_cycle_days"].mean()
stage = focus.groupby(["close_quarter", "deal_stage"]).agg(
    avg_cycle=("sales_cycle_days", "mean")
).reset_index()
stage["stall_index"] = stage.apply(lambda r: r["avg_cycle"] / overall.loc[r["close_quarter"]], axis=1)

heat = stage.pivot(index="deal_stage", columns="close_quarter", values="stall_index").reindex(
    ["Qualified", "Demo", "Proposal", "Negotiation", "Closed"]
)

mat = heat.values.astype(float)
plt.figure(figsize=(7.2, 4.8))
im = plt.imshow(mat, aspect="auto")
plt.yticks(np.arange(len(heat.index)), heat.index)
plt.xticks(np.arange(len(heat.columns)), heat.columns)
plt.title("Stage Stall Index Heatmap (Higher = Slower Than Avg)")
plt.colorbar(im, label="Stall Index")
plt.tight_layout()
plt.savefig(OUTDIR / "stage_stall_index_heatmap.png", dpi=200)
plt.close()

print("Charts saved to:", OUTDIR.resolve())
