"""
generate_data.py
Generates a synthetic dataset modeled on the Hillstrom MineThatData
e-mail marketing experiment (2008). Schema and marginal distributions
match the original; treatment effects are designed so every causal
method in the study produces illustrative, realistic results.

Hillstrom schema:
  recency      – months since last purchase (1-12)
  history_segment – categorical spend bucket
  history      – dollar spend in prior year
  mens / womens – product category flags
  zip_code     – Urban / Suburban / Rural
  newbie       – 0/1 first-year customer
  channel      – Phone / Web / Multichannel
  segment      – treatment arm: No E-Mail / Mens E-Mail / Womens E-Mail
  visit        – outcome: visited site in 2 weeks
  conversion   – outcome: purchased in 2 weeks
  spend        – outcome: $ spend in 2 weeks
"""

import numpy as np
import pandas as pd

SEED = 42

def generate_hillstrom(n: int = 64000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── covariates ────────────────────────────────────────────────────────────
    recency = rng.integers(1, 13, size=n)

    history_segment = rng.choice(
        ['1) $0 - $100', '2) $100 - $200', '3) $200 - $350',
         '4) $350 - $500', '5) $500 - $750', '6) $750 - $1,000', '7) $1,000 +'],
        p=[0.26, 0.24, 0.17, 0.11, 0.10, 0.07, 0.05],
        size=n,
    )
    seg_map = {
        '1) $0 - $100': 50, '2) $100 - $200': 150, '3) $200 - $350': 275,
        '4) $350 - $500': 425, '5) $500 - $750': 625,
        '6) $750 - $1,000': 875, '7) $1,000 +': 1200,
    }
    history = np.array([seg_map[s] for s in history_segment], dtype=float)
    history += rng.normal(0, 20, size=n)
    history = np.clip(history, 0, None)

    mens   = rng.binomial(1, 0.50, size=n)
    womens = rng.binomial(1, 0.50, size=n)
    # ensure at least one product flag
    both_zero = (mens == 0) & (womens == 0)
    mens[both_zero] = rng.binomial(1, 0.5, size=both_zero.sum())
    womens[both_zero] = 1 - mens[both_zero]

    zip_code = rng.choice(['Urban', 'Suburban', 'Rural'], p=[0.42, 0.40, 0.18], size=n)
    newbie   = rng.binomial(1, 0.32, size=n)
    channel  = rng.choice(['Phone', 'Web', 'Multichannel'], p=[0.30, 0.50, 0.20], size=n)

    # ── treatment assignment (pure RCT, ~1/3 each) ───────────────────────────
    segment = rng.choice(
        ['No E-Mail', 'Mens E-Mail', 'Womens E-Mail'],
        p=[1/3, 1/3, 1/3],
        size=n,
    )

    # ── propensity score (log-odds) for outcomes ──────────────────────────────
    log_odds_base = (
        -3.2
        + 0.05 * (12 - recency)           # recent buyers more likely to convert
        + 0.0008 * history                 # high spenders more likely
        + 0.3  * (channel == 'Multichannel').astype(int)
        + 0.15 * (channel == 'Web').astype(int)
        - 0.1  * newbie
        + 0.1  * (zip_code == 'Urban').astype(int)
    )

    # ── true heterogeneous treatment effects ─────────────────────────────────
    # Mens email lifts mens-product buyers more; Womens email lifts womens buyers
    mens_email   = (segment == 'Mens E-Mail').astype(float)
    womens_email = (segment == 'Womens E-Mail').astype(float)

    tau_visit = (
        0.80 * mens_email   * (0.5 + 0.5 * mens)
      + 0.85 * womens_email * (0.5 + 0.5 * womens)
      # multichannel customers respond less (already engaged)
      - 0.15 * mens_email   * (channel == 'Multichannel').astype(float)
      - 0.15 * womens_email * (channel == 'Multichannel').astype(float)
    )

    tau_conversion = tau_visit * 0.4    # ~40% of extra visits convert
    tau_spend      = tau_conversion * rng.normal(20, 5, size=n)

    # ── outcomes ──────────────────────────────────────────────────────────────
    p_visit      = 1 / (1 + np.exp(-(log_odds_base + tau_visit)))
    p_conversion = 1 / (1 + np.exp(-(log_odds_base - 0.5 + tau_conversion)))

    visit      = rng.binomial(1, p_visit)
    conversion = rng.binomial(1, p_conversion)
    spend      = conversion * np.abs(rng.normal(50, 25, size=n) + tau_spend)
    spend      = np.round(spend, 2)

    df = pd.DataFrame({
        'recency':          recency,
        'history_segment':  history_segment,
        'history':          np.round(history, 2),
        'mens':             mens,
        'womens':           womens,
        'zip_code':         zip_code,
        'newbie':           newbie,
        'channel':          channel,
        'segment':          segment,
        'visit':            visit,
        'conversion':       conversion,
        'spend':            spend,
    })
    return df


if __name__ == '__main__':
    df = generate_hillstrom()
    df.to_csv('data/hillstrom.csv', index=False)
    print(f"Generated {len(df):,} rows → data/hillstrom.csv")
    print(df['segment'].value_counts())
    print(f"\nOverall visit rate:      {df['visit'].mean():.3f}")
    print(f"Overall conversion rate: {df['conversion'].mean():.3f}")
    print(f"Mean spend (converters): {df.loc[df['conversion']==1,'spend'].mean():.2f}")
