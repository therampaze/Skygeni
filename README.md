# SkyGeni – Sales Intelligence Challenge (EDA + Insights)

This repository contains exploratory analysis and decision-oriented insights for a B2B SaaS sales dataset.

The CRO complaint:
> “Our win rate has dropped over the last two quarters, but pipeline volume looks healthy.”

This analysis focuses on answering:
- **What changed?**
- **Why did it change?**
- **What actions should a CRO take next?**

---

# Part 1 — Problem Framing (No Code)

## What is the real business problem?
The real problem is not only that **win rate dropped**.  
The deeper business issue is:

> **Revenue predictability and execution quality have declined, even though pipeline volume looks healthy.**

A “healthy pipeline” can still produce lower wins if:
- deal quality dropped (more bad-fit deals)
- the pipeline mix shifted (more enterprise, harder regions/industries)
- deals started stalling later in the funnel
- competition/pricing dynamics changed
- sales cycles increased, causing quarter misses

The CRO’s core need is:
> **What changed, why it changed, and what actions will reverse it quickly.**

---

## What key questions should an AI system answer for the CRO?
A useful system should answer diagnostic + decision questions:

### 1) What changed?
- Which segments drove the win-rate decline?  
  (region, industry, product type, lead source, rep)

### 2) Why did it change?
- Did pipeline mix shift toward harder-to-win deals?
- Did sales cycle increase?
- Did more deals stall in a specific stage?

### 3) Where is the current pipeline at risk?
- Which open deals resemble past losses?
- Which reps/segments show abnormal risk?

### 4) What should we do next?
- What actions will improve outcomes fastest?
- Which deals need leadership attention now?

---

## What metrics matter most for diagnosing win-rate issues?

### Outcome metrics
- Win rate (Won / Closed)
- ACV-weighted win rate
- Revenue won / lost

### Funnel health
- Stage conversion rates
- Loss rate by stage
- Stage distribution shift over time

### Velocity
- Sales cycle length
- Time in stage
- Slippage rate

### Mix + quality
- Win rate by lead source, region, industry, product type
- Pipeline composition shift over time

---

## Assumptions
- Closed deals are labeled correctly as Won/Lost
- Dates are reliable (`created_date`, `closed_date`)
- `deal_stage` represents last known stage at closure
- The dataset does not include competitor/pricing/activity signals
- “Healthy pipeline volume” means deal count and/or total ACV created is stable


---

## Dataset

Fields used:
- `deal_id`
- `created_date`
- `closed_date`
- `deal_stage`
- `deal_amount` (ACV)
- `sales_rep_id`
- `industry`
- `region`
- `product_type`
- `lead_source`
- `outcome` (won/lost)
- `sales_cycle_days`

---

## Executive Summary (What’s happening)

Comparing the last two meaningful quarters:

| Quarter | Deals Closed | Win Rate | Avg Sales Cycle | ACV Win Rate |
|--------|-------------:|---------:|----------------:|-------------:|
| 2024Q1 | 990 | 46.7% | 63.1 days | 48.1% |
| 2024Q2 | 637 | 43.8% | 80.9 days | 44.2% |

### Key signal
- Win rate dropped **~2.9 points**
- Sales cycle increased **~18 days**
- ACV-weighted win rate dropped **~3.9 points**

This strongly suggests **execution + deal velocity** issues, not pipeline volume issues.

---

## Charts

### 1) Win Rate Trend (Monthly)
![Win Rate Trend](skygeni_charts/win_rate_trend_monthly.png)

### 2) Sales Cycle Trend (Monthly)
![Sales Cycle Trend](skygeni_charts/sales_cycle_trend_monthly.png)

### 3) Win Rate by Lead Source (2024Q1 vs 2024Q2)
![Lead Source Win Rate](skygeni_charts/lead_source_win_rate_q1_q2.png)

### 4) Stage Stall Index Heatmap
![Stage Stall Index Heatmap](skygeni_charts/stage_stall_index_heatmap.png)

---

## Business Insights (Plain language + actions)

### Insight 1 — Deal velocity collapsed (sales cycle jumped)
**Finding**
- Avg sales cycle increased from **63.1 → 80.9 days**.

**Why it matters**
- Longer cycles reduce win probability (more time for competition, loss of urgency).
- It also destroys forecasting reliability.

**Action**
- Launch a **stalled deals program** (all deals > 75 days).
- Enforce mutual action plans (MAPs) and stage exit criteria.

---

### Insight 2 — Proposal stage became the main loss + stall point
**Finding**
- In 2024Q2, Proposal had the highest stall behavior and one of the lowest win rates.

**Why it matters**
- Proposal-stage losses typically indicate weak discovery, unclear value, or pricing misalignment.

**Action**
- Add a “Proposal Readiness Checklist”
- Require ROI / champion confirmation before proposal is sent.

---

### Insight 3 — Lead sources diverged (Referral improved, others worsened)
**Finding**
- Referral win rate improved in 2024Q2, while Inbound/Outbound/Partner declined.

**Why it matters**
- This is a signal of **lead quality + qualification issues**, not just sales performance.

**Action**
- Audit inbound lead scoring.
- Tighten partner deal acceptance criteria.
- Double down on referral engine (customer advocacy, partner referrals).

---

## Custom Metrics (Invented)

### Custom Metric #1 — Stage Stall Index
**Definition**
> Stage Stall Index = Avg sales cycle in stage ÷ Overall avg sales cycle

**Interpretation**
- > 1.0 = this stage is slowing deals more than expected
- < 1.0 = stage is relatively efficient

**Why it matters**
- Quickly identifies where the process is leaking time.

---

### Custom Metric #2 — Win Efficiency
**Definition**
> Win Efficiency = Total Won ACV ÷ Total Sales Cycle Days

**Interpretation**
- Measures revenue output per unit of sales time.
- Captures *both* win rate + cycle length in one number.

**Why it matters**
- Even small win-rate drops become large revenue problems when cycle time rises.

---

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

---

## Notes / Assumptions
- `sales_cycle_days` is trusted and consistent across deals.
- `deal_stage` is treated as the stage at closure (or last stage reached).
- The dataset does not include activity signals (calls/emails), competitor, or pricing changes.


---

# Part 5 — Reflection (Most Important)

## What assumptions in my solution are weakest?
The weakest assumptions are:
- **Deal stage is meaningful and consistent** across all deals  
  (in real CRMs, stages are messy and used differently by reps)
- **Sales cycle days is reliable**  
  (close dates and reopened deals can distort cycle metrics)
- **Historical won/lost patterns reflect the current market**  
  (pricing, competitors, product changes can shift outcomes quickly)
- **Lead source is correctly attributed**  
  (partner/inbound/outbound attribution is often inaccurate)

---

## What would break in real-world production?

### 1) CRM data quality issues
- Duplicate deals
- Missing close dates
- Incorrect outcomes
- Deals reopened or re-entering pipeline

### 2) Missing key predictive signals
The dataset does not include:
- calls/emails/meetings activity
- stage movement history
- CRM notes or call transcripts
- competitor, pricing, discounting, decision-maker signals

In real production, these are among the strongest predictors of deal risk.

### 3) Model drift and changing business context
- rep behavior changes
- segmentation changes
- new product launches
- macro/market shifts

Without monitoring, risk scores will become less reliable over time.

---

## What would I build next if given 1 month?
I would focus on productizing the system, not “better ML”.

### Week 1 — Data + reliability
- stage normalization
- validation checks + freshness monitoring
- incremental ingestion

### Week 2 — Add strong signals
- stage transitions + time-in-stage
- activity recency + velocity
- meeting count, email volume, touchpoints

### Week 3 — Explainability
- stable reason codes per deal
- driver analysis per segment
- CRO vs manager vs rep insight templates

### Week 4 — Delivery + workflow integration
- Slack alerts + daily risk changes
- weekly digests
- manager pipeline review queue

---

## What part of my solution am I least confident about?
I am least confident about the predictive accuracy of deal risk scoring because:
- the dataset lacks real pipeline signals
- the strongest drivers (activity + stage movement + notes) are missing
- the model is trained on closed deals but intended for open deals

So the current risk score is best viewed as a **decision-support tool**, not a perfect forecasting engine.
