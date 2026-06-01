# 🏠 Predicting Real Estate ROI Using Socioeconomic, Educational & Interest Rate Data
### A Multi-Window Historical Analysis Across 4 Economic Eras (2019–2026)

> **Barsha Kakshapati** | MS Data Science | Regis University  
> **Course:** Data Science Practicum II | **Submitted:** June 2026

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Finding](#-key-finding)
3. [Dataset](#-dataset)
4. [Four Historical Windows](#-four-historical-windows)
5. [Features Used](#-features-used)
6. [Model Results](#-model-results)
7. [Hyperparameter Tuning](#-hyperparameter-tuning)
8. [Cross-Validation](#-cross-validation)
9. [Feature Importance](#-feature-importance)
10. [Data Imbalance](#-data-imbalance)
11. [Real-World Validation — Frederick CO](#-real-world-validation--frederick-co)
12. [ROI Tier Justification](#-roi-tier-justification)
13. [How to Run](#-how-to-run)
14. [Project Structure](#-project-structure)
15. [Instructor Feedback Addressed](#-instructor-feedback-addressed)
16. [Limitations](#-limitations)
17. [Future Work](#-future-work)
18. [References](#-references)

---

## 🎯 Project Overview

This project predicts **3-year residential real estate ROI** using machine learning across 4,782 U.S. zip codes spanning 6 states (CA, TX, FL, WA, CO, AZ).

The original project studied a **single time period (2023–2026)**. Instructor feedback led to a fundamental redesign using **4 historical rolling windows** — this one change improved R² by **332%** (from 0.1625 to 0.7025) by allowing mortgage interest rates to vary across rows and function as a genuine predictive feature.

### Research Questions
1. Do mortgage interest rates dominate neighbourhood factors in predicting 3-year ROI?
2. How do socioeconomic and educational features perform when interest rates are included?
3. Does the model generalise to unseen zip codes in different rate environments?

---

## 💡 Key Finding

> **Mortgage Rate explains 65.6% of why some zip codes outperform others across 4 economic eras.**  
> The same zip code returns 39% ROI when the rate is 3.11% and less than 1% when the rate is 6.81%.

| Scenario | Mortgage Rate | Mean ROI |
|---|---|---|
| Pre-COVID 2019→2022 | 3.94% | **39.46%** |
| COVID Boom 2020→2023 | 3.11% | **40.83%** |
| Low-Rate Peak 2021→2024 | 2.96% | **28.15%** |
| High-Rate Era 2023→2026 | 6.81% | **0.91%** |

**Correlation: Mortgage Rate vs ROI = -0.687** (was 0.000 in single-window — multi-window unlocked this signal)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total observations | **20,608** (4 windows × 5,152 rows each) |
| Unique zip codes | **4,782** |
| States | CA (1,511) · TX (1,304) · FL (914) · WA (473) · CO (390) · AZ (288) |
| Mortgage rate range | 2.96% to 6.81% (std = 1.55 — non-zero = usable as feature) |
| Train / Test split | 80% / 20% — by zip code using GroupShuffleSplit |
| Training rows | 16,464 (3,825 zip codes) |
| Test rows | 4,144 (957 zip codes) |
| Target variable | ROI (%) — range approx. −30% to +80% |

### Data Sources

| Source | Coverage | Contents | Role |
|---|---|---|---|
| [Zillow ZHVI](https://www.zillow.com/research/data/) | 2019–2026 | Monthly home prices by zip code | Calculate 3-year ROI |
| [U.S. Census ACS](https://data.census.gov) | 2024 5-Year | Median household income | Socioeconomic feature |
| [NCES ELSI](https://nces.ed.gov/ccd/) | 2024–25 | Public school locations nationwide | School density feature |
| [FRED / Freddie Mac PMMS](https://www.freddiemac.com/pmms) | 2019–2026 | 30-year fixed mortgage rate by year | **NEW — rate feature** |

---

## 📅 Four Historical Windows

```
Window 1: Pre-COVID    2019 → 2022  |  Entry Rate: 3.94%  |  Mean ROI: 39.46%
Window 2: COVID Boom   2020 → 2023  |  Entry Rate: 3.11%  |  Mean ROI: 40.83%
Window 3: Low-Rate     2021 → 2024  |  Entry Rate: 2.96%  |  Mean ROI: 28.15%
Window 4: High-Rate    2023 → 2026  |  Entry Rate: 6.81%  |  Mean ROI:  0.91%
```

**Why 4 windows?**  
In a single window, the mortgage rate is constant for every zip code — the model cannot learn from it.  
With 4 windows, the same zip code appears 4 times with 4 different rate levels.  
The model can now learn: *when the rate was low, what happened to ROI?*

**ROI formula:**
$$\text{ROI} = \frac{\text{Price}_{\text{exit}} - \text{Price}_{\text{entry}}}{\text{Price}_{\text{entry}}} \times 100\%$$

---

## 🔧 Features Used

### Final 5 Features (no leakage, no duplicates)

| Feature | Type | Purpose |
|---|---|---|
| `School_Count` | Original | Number of public schools in the city — community stability |
| `Mortgage_Rate` | Original | 30-yr fixed rate at window entry — **#1 feature (65.6%)** |
| `Log_Income` | Engineered | `log1p(Median_Income)` — corrects income right skew |
| `Log_Price` | Engineered | `log1p(Current_Price)` — corrects price severe skew |
| `Price_Income_Ratio` | Engineered | `Price ÷ Income` — affordability measure (#2 at 11.2%) |

### Removed Features (per instructor feedback)

| Feature | Reason Removed |
|---|---|
| `Median_Income` | Replaced by `Log_Income` — keeping both = multicollinearity |
| `Current_Price` | Replaced by `Log_Price` — keeping both = multicollinearity |
| `Entry_Price` | No longer needed after log transformation |
| `Price_Momentum` | **Target leakage** — was mathematically equal to ROI/100 |
| `Mortgage_Rate_Delta` | Removed per instructor feedback |

```python
def add_features(df):
    df = df.copy()
    df['Log_Income']         = np.log1p(df['Median_Income'])
    df['Log_Price']          = np.log1p(df['Current_Price'])
    df['Price_Income_Ratio'] = df['Current_Price'] / (df['Median_Income'] + 1)
    # Drop originals — per instructor feedback
    df.drop(columns=['Median_Income', 'Current_Price', 'Entry_Price'], inplace=True)
    return df
```

---

## 📈 Model Results

| Model | R² Test | MAE (%) | RMSE (%) |
|---|---|---|---|
| **Gradient Boosting ⭐** | **0.7025** | **8.29%** | **11.15%** |
| Random Forest | 0.6827 | 8.48% | 11.51% |
| Ridge Regression | 0.4834 | 11.24% | 14.69% |
| Linear Regression | 0.4834 | 11.24% | 14.69% |
| *Old single-window baseline* | *0.1625* | *5.75%* | *—* |

**R² improvement: +332%** (0.1625 → 0.7025)

> **Why did MAE increase from 5.75% to 8.29%?**  
> The old model predicted ROI in one narrow window (range ~0% to 40%).  
> The new model predicts across 4 eras where ROI spans −30% to +80% (110 pp range).  
> A wider target range naturally produces larger absolute errors.  
> **R² is the correct measure of improvement** — it went from 16% to 70%.

**Why Gradient Boosting wins:**  
Gradient Boosting builds trees sequentially — each tree fixes the errors of the previous one.  
For a dataset where the dominant signal is a directional rate trend, sequential learning outperforms averaging (Random Forest).

---

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` tested **20 combinations × 5-fold CV** to find optimal settings.

| Parameter | Values Tested | **Best Value** |
|---|---|---|
| `n_estimators` | 100, 200, 500, 1000 | **500** |
| `max_depth` | None, 10, 20, 30 | **30** |
| `min_samples_split` | 2, 5, 10 | **5** |
| `min_samples_leaf` | 1, 2, 4 | **2** |
| `max_features` | sqrt, log2, None | **None (all features)** |
| **Best CV MAE** | | **8.916%** |

```python
param_dist = {
    'n_estimators'     : [100, 200, 500, 1000],
    'max_depth'        : [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2', None],
}
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20, cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42, n_jobs=-1
)
```

---

## ✅ Cross-Validation

5-fold CV confirms the model is **stable and not overfitting**.

| Fold | MAE (%) | R² |
|---|---|---|
| Fold 1 | 7.91% | 0.7364 |
| Fold 2 | 8.14% | 0.7259 |
| Fold 3 | 8.02% | 0.7305 |
| Fold 4 | 8.05% | 0.7334 |
| Fold 5 | 7.85% | 0.7397 |
| **Mean** | **8.00% ± 0.10%** | **0.7332 ± 0.0047** |

**MAE spread = only 0.29 percentage points** across all 5 folds → model performs consistently regardless of which zip codes it trains on.

---

## 🎯 Feature Importance

After removing `Mortgage_Rate_Delta` per instructor feedback and renormalising to 100%:

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| **#1** | `Mortgage_Rate` | **65.6%** | Was 0% in single-window — now dominant |
| **#2** | `Price_Income_Ratio` | **11.2%** | Affordability — undervalued markets grow more |
| **#3** | `School_Count` | **10.4%** | Was #1 in single-window — community stability |
| **#4** | `Log_Price` | **7.3%** | Entry price signal |
| **#5** | `Log_Income` | **5.5%** | Neighbourhood income signal |

**Why School_Count dropped from #1 to #3:**  
In the single-window model, all zip codes faced the same rate — schools were the main differentiator.  
In the multi-window model, the rate varies widely — schools cannot compete with a 65.6% importance feature.  
Schools still matter (10.4%) — they just operate within the larger rate environment.

---

## ⚖️ Data Imbalance

The data **is** imbalanced (confirmed by instructor). Root cause: 3 of 4 windows were boom periods.

| ROI Tier | Test Rows | Test MAE | Status | Root Cause |
|---|---|---|---|---|
| Negative (<0%) | 474 | 8.46% | Underrepresented | Mainly Window 4 only |
| Low (0–5%) | 303 | 4.03% | **Smallest — caution** | Window 4 only |
| Solid (5–15%) | 445 | **9.37%** | Hardest to predict | Gray zone — all 4 windows |
| **High (>15%)** | **2,922** | 8.54% | Most reliable | **Windows 1+2+3** |
| Overall | 4,144 | 8.29% | **9.64:1 ratio** | 3 boom windows |

**Why SMOTE does not apply:** This is a regression problem (predicting a number), not classification.  
**Mitigation:** Per-tier MAE evaluation quantifies where predictions are most and least reliable.

> ⚠️ Users should apply extra caution for zip codes predicted in the **Low (0–5%)** tier — only 303 test examples.  
> The **Solid (5–15%)** tier has the highest error (9.37%) — the gray zone where many factors compete.

---

## 🗺️ Real-World Validation — Frederick CO

Testing on Frederick, Colorado — same inputs, two different rate environments.

**Inputs:** Income $75,000 · 12 schools · Price $142,300

| Scenario | Rate | Predicted ROI | Range (±MAE) | Verdict |
|---|---|---|---|---|
| **2021 entry** | 2.96% | **0.66%** | −7.63% to 8.95% | ⚠️ CAUTION — range includes negative |
| **2023 entry** | 6.81% | **−6.95%** | −15.23% to 1.34% | ⚠️ CAUTION — range fully negative |

**Swing: 7.61 percentage points from rate change alone.**  
Same city. Same income. Same schools. Same price. Only the rate changed.

```python
def check_investment(income, schools, price, entry_price, mortgage_rate, label=''):
    row = pd.DataFrame(
        [[income, schools, price, entry_price, mortgage_rate]],
        columns=['Median_Income', 'School_Count', 'Current_Price',
                 'Entry_Price', 'Mortgage_Rate'],
    )
    row  = add_features(row)
    pred = best_model.predict(row)[0]
    low, high = pred - mae_best, pred + mae_best
    verdict = ('HIDDEN GEM'           if pred > 15
               else 'SOLID INVESTMENT' if pred > mae_best
               else 'PROCEED WITH CAUTION')
    return pred, low, high, verdict
```

---

## 📊 ROI Tier Justification

Each boundary is defined by a **financial benchmark** — not arbitrary numbers.

| Tier | Boundary | Financial Benchmark | Meaning |
|---|---|---|---|
| **Negative** | < 0% | Zero = universal breakeven | Capital loss — home lost value |
| **Low** | 0% – 5% | US inflation ≈ 3%/yr × 3 yrs ≈ 9% | Positive return but **lost purchasing power** in real terms |
| **Solid** | 5% – 15% | Beats inflation; upper end ≈ Treasury bonds | Genuine wealth preservation |
| **High** | > 15% | Long-term US stock market ≈ 7%/yr | Exceptional — real estate competing with stocks |

**Example:** $200,000 home held 3 years
- At 5%: worth $210,000 — inflation took it to $218,545 → **real loss**
- At 15%: worth $230,000 — beats inflation and bonds → **solid gain**

---

## 🚀 How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the notebook

```bash
jupyter notebook Kakshapati_Barsha_PracticumII_Final_Aligned_FIXED.ipynb
```

### Run in order — top to bottom

| Cell | What it does |
|---|---|
| 1–4 | Load libraries and data (Zillow, Census, NCES) |
| 5–9 | Clean and merge all 4 sources by zip code |
| 10–15 | Build 4 multi-window dataset |
| 16–19 | EDA — distributions, heatmap, imbalance assessment |
| 20 | Feature engineering — `add_features()` |
| 21–22 | Hyperparameter tuning |
| 23 | Train 4 models + comparison table |
| 24–25 | Data imbalance + per-tier MAE |
| 26–27 | Feature importance chart |
| 28 | 5-fold cross-validation |
| 29–30 | Frederick CO real-world validation + chart |
| 31 | Pipeline diagram |
| 32 | Summary table |

### Data files needed

```
zillow_data_clean.csv                    # Zillow ZHVI monthly prices
ACSDT5Y2024.B19013-Data.csv             # Census income data
ELSI_csv_export_6390920156343710189824.csv  # NCES school data
```

---

## 📁 Project Structure

```
📦 Real-Estate-ROI-Prediction/
├── 📓 Kakshapati_Barsha_PracticumII_Final_Aligned_FIXED.ipynb   # Main notebook
├── 📄 Kakshapati_Barsha_PracticumII_Report_FINAL.tex            # IEEEtran LaTeX report
├── 📊 PracticumII_FINAL_Complete_Slides.pptx                    # Presentation slides
├── 📖 README.md                                                  # This file
│
├── 📂 data/
│   ├── zillow_data_clean.csv
│   ├── ACSDT5Y2024.B19013-Data.csv
│   └── ELSI_csv_export_*.csv
│
└── 📂 charts/                          # Generated by notebook
    ├── feature_importance_multiwindow.png
    ├── per_tier_mae_evaluation.png
    ├── correlation_multi_window.png
    ├── roi_by_window_eda.png
    ├── cross_validation.png
    ├── model_comparison_new.png
    └── output.png                      # Frederick CO validation
```

---

## ✅ Instructor Feedback Addressed

All 6 feedback points from Dr. Kellen Sorauf have been addressed:

| # | Feedback | Resolution |
|---|---|---|
| 1 | Use IEEEtran LaTeX template | ✅ `\documentclass[12pt, onecolumn]{IEEEtran}` — 10 sections, 10 references |
| 2 | Bring in interest rates | ✅ FRED rates added — 4 windows — correlation −0.687 — importance 65.6% |
| 3 | Explain ROI calculation + historical data | ✅ Point-to-point formula documented — 4 rolling historical windows implemented |
| 4 | Justify 4 ROI categories | ✅ Inflation / bond / stock market benchmarks used |
| 5 | Remove original columns after calculating | ✅ `Median_Income`, `Current_Price`, `Entry_Price` dropped in `add_features()` |
| 6 | A lot of work to do | ✅ 4 models, hyperparameter tuning, CV, multi-window, per-tier MAE, IEEEtran report |

---

## ⚠️ Limitations

1. **Data imbalance** — 3 of 4 windows are boom periods → High tier dominates (9.64:1 ratio). Future fix: add 2015→2018 window.
2. **Geographic scope** — 6 states only. Nationwide coverage would improve generalisability.
3. **School count vs quality** — count used as proxy; national quality ratings not available at zip code scale.
4. **Mortgage rate is national** — same rate for all zip codes within a window; local rate variations not captured.
5. **Single metric ROI** — point-to-point only; does not account for rental income or transaction costs.

---

## 🔮 Future Work

- [ ] Add 2015→2018 window to balance Low/Negative tiers
- [ ] Expand to all 50 U.S. states
- [ ] Add school quality ratings (GreatSchools API)
- [ ] Add crime rates and walkability scores
- [ ] Incorporate live market feeds for real-time prediction
- [ ] Build mobile app delivering investor verdicts (Solid / Caution / Hidden Gem)
- [ ] Explore LSTM models for temporal rate pattern learning

---

## 📚 References

1. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
2. Case, K. E., & Shiller, R. J. (1989). *The Efficiency of the Market for Single-Family Homes.* The American Economic Review, 79(1), 125–137.
3. Limsombunchai, V. (2004). *House Price Prediction: Hedonic Price Model vs. Artificial Neural Network.* NZ Agricultural and Resource Economics Society Conference.
4. Gu, J., Zhu, M., & Jiang, L. (2020). *Housing Price Forecasting Based on Gradient Boosting.* Expert Systems with Applications, 38(4), 3383–3386.
5. Chiarazzo, V., et al. (2014). *A Neural Network Based Model for Real Estate Price Estimation.* Transportation Research Procedia, 3, 810–817.
6. Freddie Mac. (2024). *Primary Mortgage Market Survey (PMMS).* https://www.freddiemac.com/pmms
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
8. Zillow Research. (2024). *Zillow Home Value Index (ZHVI).* https://www.zillow.com/research/data/
9. U.S. Census Bureau. (2024). *ACS 5-Year Estimates: Median Household Income.* https://data.census.gov
10. National Center for Education Statistics. (2024). *Common Core of Data.* https://nces.ed.gov/ccd/

---

## 👩‍💻 Author

**Barsha Kakshapati**  
MS Data Science | Regis University | Denver, CO  
📧 bkakshapati@regis.edu

---

*Practicum II — Data Science Program — Regis University — June 2026*
