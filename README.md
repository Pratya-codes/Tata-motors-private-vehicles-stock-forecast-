# Tata Motors Passenger Vehicles (TMPV.NS) — Monthly Stock Price Forecasting

Forecasting Tata Motors Passenger Vehicles' monthly closing stock price (2010–2024) using
classical time-series methods — ARIMA, SARIMA, and Simple Exponential Smoothing — with a
separate GARCH(1,1) volatility analysis on returns.

## Data
- **Source:** Yahoo Finance (`yfinance`), daily close prices resampled to month-end.
- **Ticker:** `TMPV.NS`. Tata Motors Limited demerged in 2025 into Tata Motors Passenger
  Vehicles (`TMPV.NS`) and Tata Motors Commercial Vehicles (`TMCV.NS`); `TMPV.NS` was used
  since it retains the original company's historical listing data back to 2010.
- **Range:** Jan 2010 – Dec 2024, 180 monthly observations. Prices are split/dividend-adjusted
  by yfinance.

## Exploratory analysis
- **Descriptive stats:** mean ₹363.5, median ₹331.0 (right-skewed — a handful of high-price
  months near the 2024 peak pull the mean up), min ₹69.2 (COVID-19 crash), max ₹1137.4
  (2024 peak) — roughly a 16x range over the sample.
- **Visual regimes:** three distinct phases are visible — a range-bound period (2010–2015),
  a prolonged decline into the COVID-19 low (2015–2020), and a steep post-COVID rally to an
  all-time high in 2024 followed by a correction. Two apparent structural breaks (trend
  reversal ~2015–2016, steepening from the 2020 low) were noted visually; no formal
  break-point test (Chow/CUSUM/Bai-Perron) was run.
- **Stationarity (ADF test):** price levels are non-stationary (p = 0.576); first-differenced
  series is stationary well beyond the 1% threshold (p < 0.001), justifying `d=1`.
- **ACF/PACF (differenced series):** only a weak lag-1 autocorrelation (~0.17) and an isolated
  spike near lag 11; otherwise close to white noise — consistent with weak-form market
  efficiency and motivating a naive baseline as a genuine benchmark, not a token comparison.

## Methodology
1. **Chronological train / validation / test split** — train: 2010-01 to 2022-12 (156 months),
   validation: 2023 (12 months), test: 2024 (12 months). ARIMA and SARIMA hyperparameters are
   grid-searched and selected using validation error only; the test set is touched exactly
   once, after the winning model is refit on train+validation.
2. **Baseline included** — a naive (last-value) forecast is scored alongside the classical
   models so they're judged against doing nothing clever, not just against each other.
3. **GARCH(1,1)** models return volatility separately and is evaluated only against realized
   return volatility — not compared on the same RMSE scale as the price-level models, since
   volatility and price level are different units.

## Results
| Model  | Validation RMSE |
|--------|-----------------|
| SARIMA | 117.67 |
| ARIMA  | 207.32 |
| SES    | 211.91 |
| Naive  | 211.91 |

- **SARIMA** won on validation by a wide margin (~45% lower RMSE than ARIMA), indicating
  meaningful 12-month seasonal structure that non-seasonal models miss. SES converged to
  behavior nearly identical to the naive baseline, consistent with the weak short-memory
  autocorrelation seen in the ACF/PACF.
- **Test performance (SARIMA, refit on train+val):** RMSE 180.98, MAPE 16.62%. Test error is
  notably higher than validation error because the 2024 test window coincides with the
  sharpest rally-and-correction in the whole series — a regime shift no linear model trained
  on prior history could reasonably anticipate.
- **GARCH volatility forecast:** RMSE 5.22 against realized volatility ranging ~1.5–17%. The
  forecast converges toward long-run average volatility rather than tracking month-to-month
  spikes — an expected property of static multi-step GARCH forecasts, not an implementation
  error.
- **Seasonal decomposition:** multiplicative decomposition was compared against additive and
  found to be the better fit — additive residuals grow substantially in magnitude over time
  (heteroscedastic, ~±30 in 2010-12 vs ~±80-100 in 2021-24), while multiplicative residuals
  stay roughly constant (~0.7–1.3) across the full period.

## Limitations / next steps
- Single chronological split rather than full walk-forward cross-validation (short monthly
  series, ~180 points).
- No formal structural-break test — regime changes were identified visually only.
- No exogenous variables (e.g. crude oil price, given Tata Motors' auto-sector exposure).
- GARCH volatility forecasting is static/multi-step; a rolling one-step-ahead re-forecast
  would likely track realized volatility more closely.
- A daily-frequency version would allow more robust CV and additional model families (e.g.
  Prophet, LSTM) for comparison.

## How to run
```bash
pip install yfinance statsmodels arch scikit-learn matplotlib pandas
jupyter notebook TataMotors_Stock_Forecast_project.ipynb
```
