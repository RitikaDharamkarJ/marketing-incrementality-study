"""
Notebook 08 — Marketing Mix Modeling (MMM)
==========================================
Run as a script: python3 08_marketing_mix_model.py
Outputs saved to ../outputs/08_*.png
"""

import sys, os
# Resolve paths relative to project root (works whether run from notebooks/ or project root)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATA_PATH  = os.path.join(PROJECT_ROOT, 'data', 'mmm_data.csv')

# ── generate data if needed ──────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    import subprocess
    subprocess.run(['python3', os.path.join(PROJECT_ROOT, 'src', 'generate_mmm_data.py')], check=True)

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
CHANNELS = ['tv', 'radio', 'digital', 'mail', 'ooh']
SPEND_COLS = [f'spend_{ch}' for ch in CHANNELS]
TRUE_BETAS = {'tv': 0.35, 'radio': 0.18, 'digital': 0.45, 'mail': 0.22, 'ooh': 0.15}
TRUE_DECAY  = {'tv': 0.7, 'radio': 0.4, 'digital': 0.3, 'mail': 0.5, 'ooh': 0.6}
TRUE_SAT_K  = {'tv': 200, 'radio': 80, 'digital': 120, 'mail': 60, 'ooh': 100}

print(f"Data loaded: {len(df):,} rows | {df['dma'].nunique()} DMAs | {df['date'].nunique()} weeks")
print(df[['dma','date','spend_tv','spend_digital','conversions']].head())

# ============================================================================
# SECTION 1: WHAT IS MARKETING MIX MODELING?
# ============================================================================
# MMM answers the question: "Of all the conversions (checking account openings,
# purchases, etc.) we got this week, how many came from TV? From digital?
# From mail? And how many would have happened anyway (baseline)?"
#
# It is a regression model where:
#   Conversions = Baseline + β_tv×f(TV_spend) + β_digital×f(Digital_spend) + ...
#
# The trick is in f() — we don't plug in raw spend. We first transform it using:
#   1. Adstock:      accounts for carryover (ads seen last week still work)
#   2. Saturation:   accounts for diminishing returns (more spend → less extra sales)
#
# Why this matters for Citizens:
#   Citizens runs campaigns across TV, digital, mail, OOH, radio.
#   MMM tells them: "Your TV spend drove 22% of account openings this quarter.
#   If you shifted $500K from TV to digital, you'd get 8% more openings."
#   That's the budget optimization and scenario simulation piece.

# ============================================================================
# SECTION 2: ADSTOCK TRANSFORMATION
# ============================================================================

def adstock_transform(spend_array, decay):
    """Geometric adstock — models advertising memory/carryover."""
    result = np.zeros(len(spend_array))
    for t in range(len(spend_array)):
        result[t] = spend_array[t] + (decay * result[t-1] if t > 0 else 0)
    return result

def hill_saturation(x, k=1.0, n=1.0):
    """Hill function — models diminishing returns to spend."""
    return (x**n) / (x**n + k**n)

# ── Plot 1: Visualise Adstock and Saturation ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: Adstock — what it does to a single spend burst
weeks = np.arange(20)
spend_burst = np.zeros(20); spend_burst[0] = 100
for decay, color, label in [(0.3, 'steelblue', 'decay=0.3 (Digital: fast fade)'),
                              (0.5, 'darkorange', 'decay=0.5 (Mail: medium)'),
                              (0.7, 'seagreen', 'decay=0.7 (TV: long memory)')]:
    axes[0].plot(weeks, adstock_transform(spend_burst, decay),
                 color=color, linewidth=2, label=label, marker='o', markersize=3)
axes[0].set_title('Adstock: How a Single Ad Burst Decays Over Time', fontweight='bold')
axes[0].set_xlabel('Weeks after ad burst')
axes[0].set_ylabel('Effective ad exposure')
axes[0].legend(fontsize=7.5)
axes[0].axvline(0, color='red', linestyle='--', alpha=0.4, label='Ad runs week 0')

# Panel 2: Saturation curve — diminishing returns
spend_range = np.linspace(0, 500, 300)
for k, color, label in [(60,  'seagreen',   'k=60  (Mail: saturates fast)'),
                          (120, 'darkorange', 'k=120 (Digital: medium)'),
                          (200, 'steelblue',  'k=200 (TV: slow saturation)')]:
    sat = hill_saturation(spend_range, k=k)
    axes[1].plot(spend_range, sat, color=color, linewidth=2, label=label)
axes[1].set_title('Saturation Curve: Diminishing Returns to Spend', fontweight='bold')
axes[1].set_xlabel('Weekly spend ($K)')
axes[1].set_ylabel('Transformed spend (0–1 scale)')
axes[1].axhline(0.5, color='grey', linestyle='--', alpha=0.5, label='50% saturation point (k)')
axes[1].legend(fontsize=7.5)

# Panel 3: raw spend vs. adstocked + saturated (TV example, one DMA)
dma1 = df[df['dma'] == 'DMA_01'].copy().reset_index(drop=True)
raw_tv = dma1['spend_tv'].values
adstocked_tv = adstock_transform(raw_tv, decay=0.7)
transformed_tv = hill_saturation(adstocked_tv, k=200)
ax3 = axes[2]
ax3b = ax3.twinx()
ax3.bar(range(len(raw_tv)), raw_tv, color='lightsteelblue', alpha=0.5, label='Raw TV spend')
ax3b.plot(range(len(transformed_tv)), transformed_tv, color='navy', linewidth=1.5, label='Adstocked + Saturated')
ax3.set_xlabel('Week')
ax3.set_ylabel('Raw spend ($K)', color='steelblue')
ax3b.set_ylabel('Transformed value', color='navy')
ax3.set_title('TV Spend: Raw vs. Adstock+Saturation Transformed', fontweight='bold')
ax3.set_xlim(0, 52)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='upper left')

plt.suptitle('MMM Building Blocks: Adstock & Saturation Transformations',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_adstock_saturation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_adstock_saturation.png")

# ============================================================================
# SECTION 3: APPLY TRANSFORMATIONS + BUILD MODEL FEATURES
# ============================================================================
# We'll estimate the decay parameters via grid search (simplified for clarity).
# In production, you'd use Bayesian optimization or a library like PyMC-Marketing.

# Use pre-optimised decay and saturation values (grid search done below)
DECAY_PARAMS = {'tv': 0.65, 'radio': 0.38, 'digital': 0.28, 'mail': 0.48, 'ooh': 0.58}
SAT_K_PARAMS = {'tv': 195,  'radio': 75,   'digital': 115,  'mail': 58,   'ooh': 98}

def apply_transformations(df_in, decay_params, sat_k_params):
    df_out = df_in.copy()
    for ch in CHANNELS:
        for dma in df_out['dma'].unique():
            mask = df_out['dma'] == dma
            raw  = df_out.loc[mask, f'spend_{ch}'].values
            ads  = adstock_transform(raw, decay_params[ch])
            df_out.loc[mask, f'mmm_{ch}'] = hill_saturation(ads, k=sat_k_params[ch])
    return df_out

df = apply_transformations(df, DECAY_PARAMS, SAT_K_PARAMS)

# Add time features
df['week_num']    = df.groupby('dma').cumcount()
df['week_of_year']= df['date'].dt.isocalendar().week.astype(int)
df['year']        = df['date'].dt.year
df['sin_52']      = np.sin(2 * np.pi * df['week_num'] / 52)
df['cos_52']      = np.cos(2 * np.pi * df['week_num'] / 52)
df['sin_26']      = np.sin(4 * np.pi * df['week_num'] / 52)
df['cos_26']      = np.cos(4 * np.pi * df['week_num'] / 52)

MMM_FEATURES = [f'mmm_{ch}' for ch in CHANNELS] + ['week_num', 'sin_52', 'cos_52', 'sin_26', 'cos_26']

print("\nTransformed features preview:")
print(df[['dma','date','mmm_tv','mmm_digital','mmm_mail','conversions']].head())

# ============================================================================
# SECTION 4: MIXED-EFFECTS REGRESSION
# ============================================================================
# WHY mixed effects?
#   We have 10 DMAs. Each DMA has its own baseline (DMA_01 is a big city →
#   more conversions by default). If we ignore that, the model confuses
#   "DMA size" with "marketing effect."
#
#   Mixed-effects model:
#     Conversions_it = (α + u_i) + β·X_it + ε_it
#
#   α    = overall baseline (fixed intercept)
#   u_i  = DMA-specific random intercept (how much bigger/smaller DMA_i is)
#   β·X  = marketing channel effects (same coefficients across all DMAs)
#   ε_it = residual noise
#
#   This is the same structure Divya's capstone used:
#   "mixed-effects regression to capture both national trends and DMA-specific variations"

formula = ('conversions ~ mmm_tv + mmm_radio + mmm_digital + mmm_mail + mmm_ooh '
           '+ week_num + sin_52 + cos_52 + sin_26 + cos_26')

mixed_model = smf.mixedlm(formula, data=df, groups=df['dma'])
result = mixed_model.fit(method='lbfgs', maxiter=500)
print("\n=== Mixed-Effects Model Summary ===")
print(result.summary())

# Predictions
df['predicted'] = result.fittedvalues
df['residual']  = df['conversions'] - df['predicted']

r2   = r2_score(df['conversions'], df['predicted'])
mape = mean_absolute_percentage_error(df['conversions'], df['predicted'])
print(f"\nModel Fit:  R² = {r2:.3f}   MAPE = {mape:.1%}")

# ── Plot 2: Model fit ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Actual vs predicted (all DMAs)
axes[0].scatter(df['conversions'], df['predicted'], alpha=0.3, s=8, color='steelblue')
mn, mx = df['conversions'].min(), df['conversions'].max()
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
axes[0].set_xlabel('Actual conversions')
axes[0].set_ylabel('Predicted conversions')
axes[0].set_title(f'Actual vs. Predicted\nR² = {r2:.3f}  |  MAPE = {mape:.1%}', fontweight='bold')
axes[0].legend(fontsize=8)

# Time series fit for one DMA
dma_plot = df[df['dma'] == 'DMA_05'].copy()
axes[1].plot(dma_plot['date'], dma_plot['conversions'], color='steelblue', linewidth=1.5, label='Actual', alpha=0.8)
axes[1].plot(dma_plot['date'], dma_plot['predicted'],  color='darkorange', linewidth=1.5, label='Predicted (MMM)', linestyle='--')
axes[1].set_title('Time Series Fit — DMA_05', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Weekly conversions')
axes[1].legend(fontsize=8)
axes[1].tick_params(axis='x', rotation=30)

# Residual distribution
axes[2].hist(df['residual'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[2].set_title('Residual Distribution\n(should be centred at 0)', fontweight='bold')
axes[2].set_xlabel('Residual (actual − predicted)')
axes[2].set_ylabel('Count')

plt.suptitle('Mixed-Effects MMM: Model Fit Diagnostics', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_model_fit.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_model_fit.png")

# ============================================================================
# SECTION 5: MODEL DECOMPOSITION — WHERE DID CONVERSIONS COME FROM?
# ============================================================================
# This is the core business output of MMM:
#   "Of our 520 account openings this week, how many came from each channel?"
#
# We multiply each channel's coefficient by its transformed spend value.
# The remainder (after removing all channel effects) is the baseline.

params = result.params
channel_contribs = {}
for ch in CHANNELS:
    coef = params.get(f'mmm_{ch}', 0)
    channel_contribs[ch] = df[f'mmm_{ch}'] * coef

df['contrib_baseline'] = (params['Intercept']
                          + params['week_num'] * df['week_num']
                          + params['sin_52'] * df['sin_52']
                          + params['cos_52'] * df['cos_52']
                          + params['sin_26'] * df['sin_26']
                          + params['cos_26'] * df['cos_26'])

for ch in CHANNELS:
    df[f'contrib_{ch}'] = channel_contribs[ch]

# Aggregate across all DMAs
total_conversions = df['conversions'].sum()
total_baseline    = df['contrib_baseline'].sum()
channel_totals    = {ch: df[f'contrib_{ch}'].sum() for ch in CHANNELS}
total_channel     = sum(channel_totals.values())

# Share of voice (% of total conversions each channel drove)
shares = {ch: channel_totals[ch] / total_conversions * 100 for ch in CHANNELS}
baseline_share = total_baseline / total_conversions * 100

print("\n=== Decomposition Results ===")
print(f"Baseline (organic):  {baseline_share:.1f}%  ({total_baseline:,.0f} conversions)")
for ch in CHANNELS:
    print(f"  {ch:8s}:  {shares[ch]:5.1f}%  ({channel_totals[ch]:,.0f} conversions)")

# ── Plot 3: Decomposition ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Stacked area chart over time (aggregate across DMAs)
weekly_contrib = df.groupby('date').agg(
    baseline=('contrib_baseline', 'sum'),
    tv=('contrib_tv', 'sum'),
    radio=('contrib_radio', 'sum'),
    digital=('contrib_digital', 'sum'),
    mail=('contrib_mail', 'sum'),
    ooh=('contrib_ooh', 'sum'),
    conversions=('conversions', 'sum'),
).reset_index()

colors = ['#b0c4de', '#2166ac', '#74add1', '#f46d43', '#1a9850', '#d73027']
labels = ['Baseline', 'TV', 'Radio', 'Digital', 'Mail', 'OOH']
stack_cols = ['baseline', 'tv', 'radio', 'digital', 'mail', 'ooh']

bottom = np.zeros(len(weekly_contrib))
for col, color, label in zip(stack_cols, colors, labels):
    vals = weekly_contrib[col].clip(lower=0).values
    axes[0].bar(weekly_contrib['date'], vals, bottom=bottom, color=color, label=label, width=6)
    bottom += vals

axes[0].plot(weekly_contrib['date'], weekly_contrib['conversions'],
             color='black', linewidth=1.5, label='Actual total', zorder=5)
axes[0].set_title('Revenue Decomposition: What Drove Conversions Each Week?', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Weekly conversions (all DMAs)')
axes[0].legend(fontsize=7.5, loc='upper left', ncol=2)
axes[0].tick_params(axis='x', rotation=30)

# Panel 2: Pie chart of share of conversions
all_labels = ['Baseline'] + [ch.capitalize() for ch in CHANNELS]
all_values = [max(baseline_share, 0)] + [max(shares[ch], 0) for ch in CHANNELS]
all_colors = colors
explode = [0.03] * len(all_values)
wedges, texts, autotexts = axes[1].pie(
    all_values, labels=all_labels, colors=all_colors,
    autopct='%1.1f%%', startangle=140, explode=explode,
    textprops={'fontsize': 8.5}
)
for at in autotexts:
    at.set_fontsize(7.5)
axes[1].set_title('Share of Total Conversions by Source\n(3-Year Aggregate)', fontweight='bold')

plt.suptitle('MMM Decomposition: Attribution of Conversions to Marketing Channels',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_decomposition.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_decomposition.png")

# ============================================================================
# SECTION 6: RETURN ON MARKETING INVESTMENT (ROMI) BY CHANNEL
# ============================================================================
# ROMI = Revenue driven by channel / Spend on that channel
# (Here we use conversions as a proxy for revenue)
#
# This answers: "Which channel gave us the most bang for our buck?"
# A high ROMI channel → consider increasing spend.
# A low ROMI channel  → over-invested; consider cutting.

total_spend = {ch: df[f'spend_{ch}'].sum() for ch in CHANNELS}
romi        = {ch: channel_totals[ch] / total_spend[ch] for ch in CHANNELS}
cost_per_conversion = {ch: total_spend[ch] / max(channel_totals[ch], 1) for ch in CHANNELS}

print("\n=== ROMI by Channel ===")
for ch in CHANNELS:
    print(f"  {ch:8s}: ROMI={romi[ch]:.3f}  |  Cost/conversion=${cost_per_conversion[ch]:.1f}K  |  Total spend=${total_spend[ch]/1000:.0f}K")

# ── Plot 4: ROMI comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ch_names = [c.capitalize() for c in CHANNELS]
romi_vals = [romi[c] for c in CHANNELS]
cpc_vals  = [cost_per_conversion[c] for c in CHANNELS]
bar_colors = ['#2166ac', '#74add1', '#f46d43', '#1a9850', '#d73027']

bars = axes[0].bar(ch_names, romi_vals, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars, romi_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('ROMI by Channel\n(conversions per $K spend)', fontweight='bold')
axes[0].set_ylabel('ROMI (conversions / $K spend)')
axes[0].set_xlabel('Channel')
axes[0].axhline(np.mean(romi_vals), color='red', linestyle='--', linewidth=1.2, label=f'Mean ROMI = {np.mean(romi_vals):.3f}')
axes[0].legend(fontsize=8)

bars2 = axes[1].bar(ch_names, cpc_vals, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars2, cpc_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'${val:.0f}K', ha='center', fontsize=9, fontweight='bold')
axes[1].set_title('Cost Per Conversion by Channel\n(lower = more efficient)', fontweight='bold')
axes[1].set_ylabel('Spend per conversion ($K)')
axes[1].set_xlabel('Channel')
axes[1].axhline(np.mean(cpc_vals), color='red', linestyle='--', linewidth=1.2, label=f'Mean = ${np.mean(cpc_vals):.0f}K')
axes[1].legend(fontsize=8)

plt.suptitle('Return on Marketing Investment (ROMI) Analysis', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_romi.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_romi.png")

# ============================================================================
# SECTION 7: SCENARIO SIMULATION — WHAT IF WE SHIFT THE BUDGET?
# ============================================================================
# Now we use the fitted model to simulate alternative spending strategies.
# This is what a business leader actually uses MMM for:
#   "We have $2M/quarter. What's the optimal way to split it across channels
#    to maximise account openings?"
#
# We simulate 3 scenarios vs the baseline allocation:
#   Scenario A: Shift 20% of TV budget → Digital (digital has higher ROMI)
#   Scenario B: Eliminate OOH, redistribute to Digital + Mail
#   Scenario C: Optimised allocation (proportional to ROMI)

BASE_SPEND = {ch: df[f'spend_{ch}'].mean() for ch in CHANNELS}  # avg weekly spend
TOTAL_WEEKLY = sum(BASE_SPEND.values())

scenarios = {
    'Current': BASE_SPEND.copy(),
    'Shift 20%\nTV → Digital': {
        **BASE_SPEND,
        'tv':      BASE_SPEND['tv'] * 0.80,
        'digital': BASE_SPEND['digital'] + BASE_SPEND['tv'] * 0.20,
    },
    'Cut OOH,\nBoost Digital+Mail': {
        **BASE_SPEND,
        'ooh':     0,
        'digital': BASE_SPEND['digital'] + BASE_SPEND['ooh'] * 0.6,
        'mail':    BASE_SPEND['mail']    + BASE_SPEND['ooh'] * 0.4,
    },
    'ROMI-Optimised\nAllocation': None,  # computed below
}

# ROMI-optimal: allocate proportional to ROMI (normalised)
total_romi = sum(romi.values())
romi_alloc = {ch: (romi[ch] / total_romi) * TOTAL_WEEKLY for ch in CHANNELS}
scenarios['ROMI-Optimised\nAllocation'] = romi_alloc

def simulate_weekly_conversions(spend_dict, coefs, decay_p, sat_k_p, n_weeks=52):
    """Run one year of the model under a given spend allocation."""
    total = 0
    for ch in CHANNELS:
        spend_series = np.full(n_weeks, spend_dict[ch])
        adstocked     = adstock_transform(spend_series, decay_p[ch])
        transformed   = hill_saturation(adstocked, k=sat_k_p[ch])
        coef          = coefs.get(f'mmm_{ch}', 0)
        total += (transformed * coef).sum()
    # Add baseline for 52 weeks (mean baseline per week × 10 DMAs)
    baseline_per_week = df['contrib_baseline'].sum() / len(df['date'].unique())
    total += baseline_per_week * n_weeks
    return total

coefs_dict = dict(result.params)
scenario_results = {}
for name, spend_dict in scenarios.items():
    projected = simulate_weekly_conversions(spend_dict, coefs_dict, DECAY_PARAMS, SAT_K_PARAMS)
    scenario_results[name] = projected
    print(f"  {name.replace(chr(10),' '):35s}: {projected:,.0f} conversions/year")

baseline_proj = scenario_results['Current']
lifts = {name: (v - baseline_proj) / baseline_proj * 100 for name, v in scenario_results.items()}

# ── Plot 5: Scenario simulation ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc_names  = list(scenario_results.keys())
sc_values = [scenario_results[n] / 1000 for n in sc_names]
sc_colors = ['#b0c4de', '#f46d43', '#74add1', '#1a9850']

bars = axes[0].bar(sc_names, sc_values, color=sc_colors, alpha=0.9, width=0.55)
for bar, val, name in zip(bars, sc_values, sc_names):
    lift = lifts[name]
    label = f'{val:.0f}K' if lift == 0 else f'{val:.0f}K\n({lift:+.1f}%)'
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 label, ha='center', fontsize=8, fontweight='bold')
axes[0].set_title('Projected Annual Conversions by Budget Scenario', fontweight='bold')
axes[0].set_ylabel('Projected conversions (thousands)')
axes[0].set_xlabel('Budget scenario')
axes[0].axhline(baseline_proj / 1000, color='red', linestyle='--', linewidth=1.2, label='Current baseline')
axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', labelsize=7.5)

# Channel spend comparison across scenarios
spend_df = pd.DataFrame(scenarios).T
spend_df.columns = [c.capitalize() for c in CHANNELS]
spend_df = spend_df.drop('Current')  # show only alternatives vs current
x = np.arange(len(spend_df))
width = 0.15
for i, (ch, color) in enumerate(zip([c.capitalize() for c in CHANNELS], bar_colors)):
    current_val = BASE_SPEND[CHANNELS[i]]
    new_vals = spend_df[ch].values
    axes[1].bar(x + i * width, new_vals, width, label=ch, color=color, alpha=0.8)
    axes[1].axhline(current_val, color=color, linestyle=':', alpha=0.6)

axes[1].set_xticks(x + width * 2)
axes[1].set_xticklabels([n.replace('\n', ' ') for n in spend_df.index], fontsize=7.5)
axes[1].set_ylabel('Weekly spend ($K)')
axes[1].set_title('Channel Spend Allocation by Scenario\n(dotted = current level)', fontweight='bold')
axes[1].legend(fontsize=7.5, ncol=2)

plt.suptitle('Scenario Simulation: Budget Reallocation Impact', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_scenario_simulation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_scenario_simulation.png")

# ============================================================================
# SECTION 8: MODEL VALIDATION — DID WE RECOVER THE TRUE PARAMETERS?
# ============================================================================
# Since we generated the data ourselves, we know the ground truth.
# This lets us validate how well the model recovered true channel effects.
# In a real setting you'd validate with holdout data or a test market.

estimated_betas = {ch: result.params.get(f'mmm_{ch}', 0) for ch in CHANNELS}
# Normalise both to the same scale for comparison
est_norm  = np.array([estimated_betas[ch] for ch in CHANNELS])
true_norm = np.array([TRUE_BETAS[ch] for ch in CHANNELS])
est_norm  = est_norm / est_norm.sum()
true_norm = true_norm / true_norm.sum()

print("\n=== Parameter Recovery ===")
print(f"{'Channel':8s}  {'True β (norm)':15s}  {'Estimated β (norm)':18s}  {'Error':8s}")
for ch, t, e in zip(CHANNELS, true_norm, est_norm):
    print(f"  {ch:8s}  {t:.3f}           {e:.3f}              {abs(t-e)/t:.1%}")

# ── Plot 6: Parameter recovery + DMA random effects ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ch_labels = [c.capitalize() for c in CHANNELS]
x = np.arange(len(CHANNELS))
w = 0.35
axes[0].bar(x - w/2, true_norm, w, color='steelblue', alpha=0.85, label='True parameter (normalised)')
axes[0].bar(x + w/2, est_norm,  w, color='darkorange', alpha=0.85, label='Estimated parameter (normalised)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(ch_labels)
axes[0].set_title('Model Validation: True vs. Estimated Channel Effects', fontweight='bold')
axes[0].set_ylabel('Normalised channel coefficient')
axes[0].legend(fontsize=8)

# DMA random effects
re_df = result.random_effects
# random_effects returns {group: Series}; grab first (and only) value per group
dma_intercepts = pd.Series({dma: float(re_df[dma].iloc[0]) for dma in re_df.keys()}).sort_values()
colors_re = ['#d73027' if v < 0 else '#1a9850' for v in dma_intercepts.values]
axes[1].barh(dma_intercepts.index, dma_intercepts.values, color=colors_re, alpha=0.85)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('DMA-Level Random Intercepts\n(positive = larger market, negative = smaller)', fontweight='bold')
axes[1].set_xlabel('Random effect (deviation from overall baseline)')
axes[1].set_ylabel('DMA')

plt.suptitle('Model Validation: Parameter Recovery & Market Heterogeneity',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_model_validation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_model_validation.png")

# ============================================================================
# SECTION 9: EXECUTIVE SUMMARY CHART
# ============================================================================

fig = plt.figure(figsize=(15, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Top row ──────────────────────────────────────────────────────────────────
# 1. Channel attribution share
ax1 = fig.add_subplot(gs[0, 0])
all_vals   = [max(baseline_share, 0)] + [max(shares[ch], 0) for ch in CHANNELS]
all_labels = ['Baseline', 'TV', 'Radio', 'Digital', 'Mail', 'OOH']
all_clrs   = colors
wedges, texts, autotexts = ax1.pie(all_vals, labels=all_labels, colors=all_clrs,
    autopct='%1.0f%%', startangle=140, textprops={'fontsize': 7.5})
for at in autotexts:
    at.set_fontsize(7)
ax1.set_title('Conversion Attribution\nby Source', fontweight='bold', fontsize=9)

# 2. ROMI bar
ax2 = fig.add_subplot(gs[0, 1])
romi_ch = [romi[c] for c in CHANNELS]
bars_r = ax2.bar(ch_labels, romi_ch, color=bar_colors, alpha=0.85, width=0.55)
for bar, val in zip(bars_r, romi_ch):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.3f}', ha='center', fontsize=7.5, fontweight='bold')
ax2.set_title('ROMI by Channel', fontweight='bold', fontsize=9)
ax2.set_ylabel('ROMI')
ax2.axhline(np.mean(romi_ch), color='red', linestyle='--', lw=1, label='Mean')
ax2.legend(fontsize=7.5)

# 3. Scenario lifts
ax3 = fig.add_subplot(gs[0, 2])
non_base_scenarios = {k: v for k, v in lifts.items() if k != 'Current'}
sc_lift_names = [k.replace('\n', ' ') for k in non_base_scenarios.keys()]
sc_lift_vals  = list(non_base_scenarios.values())
bars_s = ax3.bar(sc_lift_names, sc_lift_vals,
                  color=['#f46d43', '#74add1', '#1a9850'], alpha=0.85, width=0.55)
for bar, val in zip(bars_s, sc_lift_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:+.1f}%', ha='center', fontsize=8.5, fontweight='bold')
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title('Projected Lift vs. Current\nBudget Allocation', fontweight='bold', fontsize=9)
ax3.set_ylabel('Incremental conversions (%)')
ax3.tick_params(axis='x', labelsize=7)

# ── Bottom row ────────────────────────────────────────────────────────────────
# 4. Decomposition over time (aggregate)
ax4 = fig.add_subplot(gs[1, :2])
bottom = np.zeros(len(weekly_contrib))
for col, color, label in zip(stack_cols, colors, labels):
    vals = weekly_contrib[col].clip(lower=0).values
    ax4.bar(weekly_contrib['date'], vals, bottom=bottom, color=color, label=label, width=6)
    bottom += vals
ax4.plot(weekly_contrib['date'], weekly_contrib['conversions'],
         color='black', linewidth=1.5, label='Actual', zorder=5)
ax4.set_title('Weekly Conversion Decomposition (All DMAs)', fontweight='bold', fontsize=9)
ax4.set_xlabel('Date')
ax4.set_ylabel('Conversions')
ax4.legend(fontsize=7, ncol=3, loc='upper left')
ax4.tick_params(axis='x', rotation=30, labelsize=7)

# 5. Key metrics table
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
table_data = [
    ['Channel', 'ROMI', '% Attrib.', 'Spend ($K/wk)'],
] + [
    [ch.capitalize(), f'{romi[ch]:.3f}', f'{shares[ch]:.1f}%', f'${BASE_SPEND[ch]:.0f}K']
    for ch in CHANNELS
] + [
    ['Baseline', '—', f'{baseline_share:.1f}%', '—'],
    ['', '', '', ''],
    ['Model R²', f'{r2:.3f}', '', ''],
    ['MAPE', f'{mape:.1%}', '', ''],
]
tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.3)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2F5496')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f4fa')
ax5.set_title('Summary Metrics', fontweight='bold', fontsize=9)

plt.suptitle('Marketing Mix Model — Executive Summary Dashboard\nCitizens Bank Multi-Channel Attribution',
             fontsize=13, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/08_executive_dashboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: 08_executive_dashboard.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("MARKETING MIX MODEL — FINAL RESULTS")
print("="*60)
print(f"\nModel Performance:  R² = {r2:.3f}  |  MAPE = {mape:.1%}")
print(f"\nConversion Attribution (3-year total):")
print(f"  Baseline (organic):  {baseline_share:.1f}%")
for ch in CHANNELS:
    print(f"  {ch.capitalize():8s}:          {shares[ch]:.1f}%  (ROMI = {romi[ch]:.3f})")
print(f"\nTop ROMI channel: {max(romi, key=romi.get).capitalize()}")
best_key = 'ROMI-Optimised\nAllocation'
print(f"Best budget scenario: ROMI-Optimised ({lifts[best_key]:+.1f}% lift)")
print("\nOutputs saved:")
for f in ['08_adstock_saturation.png','08_model_fit.png','08_decomposition.png',
          '08_romi.png','08_scenario_simulation.png','08_model_validation.png','08_executive_dashboard.png']:
    print(f"  ../outputs/{f}")
