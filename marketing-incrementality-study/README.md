# Marketing Incrementality & Causal Impact Study

**Python • SQL • Propensity Methods • Difference-in-Differences • Uplift Modeling • Treatment Effects**

A reproducible, end-to-end marketing measurement workflow that estimates the *incremental* causal impact of an email campaign — separating true lift from organic conversion — using multiple causal inference methods.

---

## Project Summary

Naive attribution (crediting all conversions to the campaign) systematically overstates ROI. This project applies rigorous causal methods to estimate the true incremental effect of a marketing campaign, compare estimates across methods, and translate findings into concrete budget and targeting recommendations.

**Dataset:** Hillstrom MineThatData E-Mail Analytics Challenge — a three-arm randomized controlled trial with 64,000 customers, two treatment arms (Mens Email, Womens Email), and outcomes measured 2 weeks post-campaign (site visit, conversion, spend).

---

## Methods & Notebooks

| Notebook | Method | Key Output |
|---|---|---|
| `01_eda_and_data_prep` | EDA, balance checks, feature engineering | Covariate distributions, SMD balance chart |
| `02_naive_attribution` | Simple mean difference; propensity confounding demo | Why naive attribution is biased |
| `03_difference_in_differences` | DiD via OLS; parallel trends check | ATE with heterogeneous effects by channel |
| `04_propensity_score_matching` | PSM (nearest-neighbour, caliper = 0.05) | Pre/post-match balance; ATT on matched sample |
| `05_uplift_modeling` | T-Learner (GBM); Qini curve; decile analysis | CATE distribution; targeting efficiency chart |
| `06_sensitivity_and_diagnostics` | Placebo test; bootstrap CIs; effect stability | Confidence bounds; decay-over-time analysis |
| `07_executive_summary` | ROI calculation; targeting recommendation | Budget recommendation for stakeholders |

---

## Key Findings

- **Naive ROI overstated causal ROI by ~3×** — crediting organic converters inflates campaign value
- **Causal lift is real and statistically significant** across DiD, PSM, and uplift methods
- **Mens buyers on Web channel** show highest individual-level uplift (heterogeneous effects)
- **Multichannel customers** show lower incremental lift — already engaged, suppressing them saves budget
- **Targeting top 50% by predicted uplift** improves campaign ROI while reducing spend ~50%
- **Effect monitoring** shows lift decays ~15%/week post-campaign — justifies retargeting cadence

---

## Project Structure

```
marketing-incrementality-study/
├── README.md
├── requirements.txt
├── data/
│   ├── hillstrom.csv          # Raw dataset (generated from src/generate_data.py)
│   └── hillstrom_clean.csv    # Feature-engineered dataset
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb
│   ├── 02_naive_attribution.ipynb
│   ├── 03_difference_in_differences.ipynb
│   ├── 04_propensity_score_matching.ipynb
│   ├── 05_uplift_modeling.ipynb
│   ├── 06_sensitivity_and_diagnostics.ipynb
│   └── 07_executive_summary.ipynb
├── src/
│   └── generate_data.py       # Reproducible data generation (Hillstrom schema)
└── outputs/
    └── *.png                  # All charts saved here
```

---

## Quick Start

```bash
git clone https://github.com/RitikaDharamkarJ/marketing-incrementality-study
cd marketing-incrementality-study
pip install -r requirements.txt

# Generate data
python src/generate_data.py

# Run notebooks in order (1 → 7)
jupyter notebook notebooks/
```

---

## Causal Methods Reference

| Method | Identifies | Assumption |
|---|---|---|
| Difference-in-Differences | ATE / ATT | Parallel trends (pre-period) |
| Propensity Score Matching | ATT | No hidden confounders; common support |
| T-Learner Uplift | CATE (individual) | No hidden confounders; model correct |
| Placebo / permutation test | Robustness check | Effect = 0 under null |

---

## Skills Demonstrated

- Causal inference: DiD, PSM, uplift modeling, heterogeneous treatment effects
- Python: scikit-learn, statsmodels, pandas, matplotlib, seaborn
- SQL-compatible data workflows (panel reshaping, aggregations)
- Model monitoring and sensitivity analysis
- Executive communication: translating statistical results to ROI and budget recommendations

---

*Dataset based on the Hillstrom MineThatData E-Mail Analytics Challenge (K. Hillstrom, 2008). Synthetic data generated with matched distributional properties for reproducibility.*
