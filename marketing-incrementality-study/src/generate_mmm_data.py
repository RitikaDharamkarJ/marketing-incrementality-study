"""
generate_mmm_data.py
--------------------
Generates synthetic weekly marketing spend + sales data for Marketing Mix Modeling.

Design:
  - 3 years of weekly data (156 weeks) across 10 DMAs (media markets)
  - 5 channels: TV, Radio, Digital, Mail, OOH (out-of-home)
  - True channel effects with Adstock (carryover + diminishing returns)
  - Baseline trend, seasonality, macroeconomic noise, holiday spikes
  - Ground truth stored so we can evaluate model recovery
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── config ──────────────────────────────────────────────────────────────────
N_WEEKS = 156          # 3 years
N_DMAS  = 10
DMAS    = [f'DMA_{i:02d}' for i in range(1, N_DMAS + 1)]

# True channel parameters (what the model should recover)
CHANNELS = {
    'tv':      {'decay': 0.7,  'saturation_k': 200, 'true_beta': 0.35},
    'radio':   {'decay': 0.4,  'saturation_k': 80,  'true_beta': 0.18},
    'digital': {'decay': 0.3,  'saturation_k': 120, 'true_beta': 0.45},
    'mail':    {'decay': 0.5,  'saturation_k': 60,  'true_beta': 0.22},
    'ooh':     {'decay': 0.6,  'saturation_k': 100, 'true_beta': 0.15},
}

BASELINE_CONVERSIONS = 500   # weekly baseline per DMA
TREND_SLOPE          = 0.8   # slight upward trend per week
NOISE_SD             = 30    # residual noise

def adstock(spend_series, decay):
    """
    Adstock transformation: accounts for the carryover effect of advertising.

    Intuition: A TV ad seen this week still influences behaviour next week and
    the week after — just at a diminishing rate. The decay parameter (between
    0 and 1) controls how fast that memory fades.

    decay=0.7 → 70% of this week's impact carries into next week.
    decay=0.3 → rapid decay; effect is mostly gone within 1–2 weeks.
    """
    adstocked = np.zeros(len(spend_series))
    for t, spend in enumerate(spend_series):
        if t == 0:
            adstocked[t] = spend
        else:
            adstocked[t] = spend + decay * adstocked[t - 1]
    return adstocked


def hill_saturation(x, k):
    """
    Hill / diminishing-returns transformation.

    Intuition: Doubling your TV spend doesn't double your sales.
    The first $100K of TV has high impact; the next $100K has less.
    k is the 'half-saturation point' — spend level at 50% of max effect.
    """
    return x / (x + k)


def generate_dma_data(dma_id, dma_scale):
    """Generate weekly data for one DMA."""
    weeks = np.arange(N_WEEKS)

    # ── spend per channel (realistic patterns) ──────────────────────────────
    spend = {}
    spend['tv']      = np.abs(np.random.normal(180, 40, N_WEEKS)) * dma_scale
    spend['radio']   = np.abs(np.random.normal(70,  20, N_WEEKS)) * dma_scale
    spend['digital'] = np.abs(np.random.normal(100, 30, N_WEEKS)) * dma_scale
    spend['mail']    = np.zeros(N_WEEKS)  # mail is periodic (every 6 weeks)
    spend['ooh']     = np.abs(np.random.normal(90,  25, N_WEEKS)) * dma_scale

    # Mail drops every 6 weeks — a discrete campaign cadence
    for w in range(0, N_WEEKS, 6):
        spend['mail'][w] = np.abs(np.random.normal(50, 10)) * dma_scale

    # Q4 / holiday TV boost (weeks 44–52, 96–104, 148–156)
    for holiday_start in [44, 96, 148]:
        end = min(holiday_start + 8, N_WEEKS)
        spend['tv'][holiday_start:end] *= 1.6
        spend['digital'][holiday_start:end] *= 1.3

    # ── apply adstock + saturation ───────────────────────────────────────────
    transformed = {}
    for ch, params in CHANNELS.items():
        adstocked = adstock(spend[ch], params['decay'])
        transformed[ch] = hill_saturation(adstocked, params['saturation_k'])

    # ── build conversions (ground truth) ────────────────────────────────────
    baseline    = BASELINE_CONVERSIONS * dma_scale
    trend       = TREND_SLOPE * weeks
    seasonality = 40 * np.sin(2 * np.pi * weeks / 52) + 20 * np.sin(4 * np.pi * weeks / 52)
    noise       = np.random.normal(0, NOISE_SD, N_WEEKS)

    channel_effect = sum(
        CHANNELS[ch]['true_beta'] * transformed[ch] * BASELINE_CONVERSIONS
        for ch in CHANNELS
    )

    conversions = baseline + trend + seasonality + channel_effect + noise
    conversions = np.maximum(conversions, 0)

    # ── assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({'week': weeks, 'dma': dma_id})
    df['date'] = pd.date_range('2022-01-03', periods=N_WEEKS, freq='W-MON')
    for ch in CHANNELS:
        df[f'spend_{ch}'] = spend[ch].round(1)
    for ch in CHANNELS:
        df[f'adstock_{ch}']  = adstock(spend[ch], CHANNELS[ch]['decay']).round(4)
        df[f'transformed_{ch}'] = transformed[ch].round(4)
    df['baseline']       = (baseline + trend + seasonality).round(1)
    df['channel_effect'] = channel_effect.round(1)
    df['conversions']    = conversions.round(0).astype(int)
    df['total_spend']    = sum(spend[ch] for ch in CHANNELS).round(1)
    df['dma_scale']      = dma_scale
    return df


def main():
    # DMA size multipliers (large vs small markets)
    dma_scales = np.linspace(0.6, 1.4, N_DMAS)
    frames = []
    for i, dma in enumerate(DMAS):
        frames.append(generate_dma_data(dma, dma_scales[i]))

    data = pd.concat(frames, ignore_index=True)
    data.to_csv('/home/claude/marketing-incrementality-study/data/mmm_data.csv', index=False)
    print(f"MMM data saved: {len(data):,} rows  ({N_DMAS} DMAs × {N_WEEKS} weeks)")
    print(data[['dma','date','spend_tv','spend_digital','spend_mail','conversions']].head(10))
    return data


if __name__ == '__main__':
    main()
