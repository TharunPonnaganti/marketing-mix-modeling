# Marketing Mix Model (MMM)

An end-to-end, interpretable marketing mix model in Python. It takes weekly
media spend by channel and a sales (revenue) series, and estimates how much
each channel contributed to sales and the ROI of each channel, with holdout
validation.

```
spend by channel  ──adstock──▶  ──saturation──▶  joint regression  ──▶  sales
                                                         │
                                 channel contribution decomposition + ROI
```

## Why these steps

**Adstock (carryover).** Advertising keeps working after the week it runs, so
each channel's spend is passed through a geometric decay
`a_t = x_t + decay · a_{t-1}`. The decay rate is a hyperparameter, tuned per
channel by validation fit.

**Saturation (diminishing returns).** Doubling spend does not double response,
so adstocked spend is passed through a Hill curve
`x^s / (x^s + half^s)`. The half-saturation point is set above the mean so the
model operates on the informative part of the curve.

**Joint base + media fit.** Base demand (intercept, linear trend, and yearly
seasonality via Fourier terms) and the transformed media channels are estimated
**together** in one bounded least-squares fit, with a **non-negativity
constraint on the media coefficients only** (media cannot subtract from sales).
Fitting them jointly is what stops the trend/seasonality terms from absorbing
the media signal — the single most common way a naive MMM fails.

**Decomposition & ROI.** Each channel's weekly contribution is
`coefficient × transformed spend`. Channel ROI is
`total contributed revenue / total spend`.

**Validation.** A time-based 80/20 split: the model is fit on the first 80% of
weeks and scored on the held-out last 20% (R² and MAPE), so the fit is judged
on weeks it never saw.

## Run

```bash
pip install -r requirements.txt
python mmm.py
```

Outputs:
- console summary (train/holdout R² and MAPE, channel ROI table, adstock decays)
- `mmm_results.csv` — per-channel spend, contribution, ROI, tuned adstock decay
- `outputs/mmm_fit.png` — actual vs predicted weekly sales, with the holdout boundary
- `outputs/mmm_contributions.png` — stacked base + channel contribution over time
- `outputs/mmm_roi.png` — channel ROI bar chart

## Data

By default the script **simulates** a realistic 3-year weekly dataset with a
known ground truth (true adstock decay and true ROI per channel), then prints
the recovered ROI next to the true ROI as a sanity check — a way to show the
estimator is unbiased in the right direction before trusting it on real data.

To run on real data, replace `simulate_data()` with `load_csv(path)`; the CSV
needs a `date` column, a `sales` column, and one `<channel>_spend` column per
channel.

## Notes & honest limitations

- Recovered ROIs track the true ROIs in rank and magnitude but are not exact —
  that is expected: adstock, saturation, and collinearity make MMM an
  approximation, which is why results are reported as directional ROI with
  validation, not point truth.
- This is a frequentist, point-estimate MMM. Production systems often add
  Bayesian priors (e.g., PyMC-Marketing / Robyn), confidence intervals, and
  budget-optimization on top of the same transforms used here.
