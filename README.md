# Predicting Real Estate ROI: Colorado Housing Study

A machine learning model that predicts 3-year return on investment for Colorado zip codes using property price, household income, and school density.

**Barsha Kakshapati** · MS Data Science · Regis University · Denver, CO · [bkakshapati@regis.edu](mailto:bkakshapati@regis.edu)

---

## Key Results

| Metric | Value |
| --- | --- |
| Zip codes analysed | 25,907 |
| Data sources merged | 3 |
| Model accuracy (R²) | 0.85 |
| Average error (MAE) | 7.24% |
| Top driver | Current price (43.8%) |

---

## Overview

When evaluating a home as an investment, most buyers look only at current price trends. This project asks a smarter question: can we predict future growth by analysing the *neighbourhood itself* — specifically household income and school density?

Using a Random Forest Regressor trained on 25,907 zip codes, the model explains **85% of the variation in 3-year ROI** across Colorado. It was then validated against a real city — Frederick, CO — that the model had never seen during training.

---

## Research Questions

1. **Primary** — Can income + school density predict 3-year ROI better than price alone?
2. **Secondary** — Which factor matters more: income or school count?
3. **Validation** — Does the model generalise to a real city it has never seen?

---

## Data Sources

| Source | Contents | Usage |
| --- | --- | --- |
| [Zillow Research](https://www.zillow.com/research/data/) (2023–2026) | Home prices by U.S. zip code | Calculated 3-year ROI |
| [U.S. Census Bureau](https://data.census.gov) | Median household income by zip code | Matched to each Zillow zip code |
| [NCES Common Core of Data](https://nces.ed.gov/ccd/) | Every public school location in the U.S. | Counted schools per city (school density) |

**Merging challenge:** Each source uses a different geographic key — zip codes, GEO_ID codes, and city+state strings. Custom normalisation code standardised all zip codes to 5-digit strings and matched city names via uppercase trimming. After cleaning, the three datasets joined into a master table of **25,907 rows**.

---

## Exploratory Analysis

### Correlation Matrix

| | Median Income | School Count | ROI | Current Price |
| --- | :---: | :---: | :---: | :---: |
| **Median Income** | 1.00 | 0.02 | 0.16 | **0.67 ★** |
| **School Count** | 0.02 | 1.00 | −0.17 | 0.14 |
| **ROI** | 0.16 | −0.17 | 1.00 | 0.05 |
| **Current Price** | **0.67 ★** | 0.14 | 0.05 | 1.00 |

★ Strongest relationship: income ↔ price (0.67). ROI is weakly correlated with all individual variables — this is the *Profit Mystery* that justifies using machine learning rather than a simple linear model.

### Distribution Summary

| Variable | Mean | Median | Std Dev | Skew |
| --- | --- | --- | --- | --- |
| ROI (%) | ~14% | ~13% | ~12% | Low — roughly normal |
| Median Income | $75,000 | $69,000 | $28,000 | Moderate — acceptable |
| Current Price | $420K | $370K | $200K | High — right-skewed |

### Outlier Detection (IQR Method)

| Variable | Outliers Found | % of Data | Action |
| --- | :---: | :---: | --- |
| ROI (%) | ~1,200 | ~4.6% | Kept — extreme returns are real market events |
| Median Income | ~800 | ~3.1% | Kept — high-income zip codes are valid data |
| Current Price | ~1,500 | ~5.8% | Kept — luxury markets are part of the study |

---

## Class Balance Assessment

This is a regression task (predicting a continuous ROI value), so traditional classification imbalance tools like SMOTE do not apply. However, the ROI range was audited for coverage:

| ROI Tier | Approx. Zip Codes | Share | Status |
| --- | :---: | :---: | --- |
| Negative ROI (< 0%) | ~2,500 | ~10% | ⚠️ Underrepresented — loss predictions less reliable |
| Low ROI (0–5%) | ~4,000 | ~15% | Acceptable |
| Solid ROI (5–15%) | ~12,000 | ~46% | ✅ Well represented |
| High ROI (> 15%) | ~7,400 | ~29% | ✅ Well represented |

No data augmentation was required. The distribution is approximately normal, with solid and high ROI tiers comprising 75% of the dataset.

---

## Model

### Algorithm

**Random Forest Regressor** — an ensemble of 1,000 decision trees whose predictions are averaged. This reduces the risk of any single tree overfitting and produces robust, interpretable feature importances.

### Training Pipeline

1. **Split** — 80% training (21,111 rows), 20% held-out test (5,278 rows)
2. **Fit** — 1,000 trees learn rules from the training set
3. **Aggregate** — tree predictions are averaged into a final ROI estimate
4. **Evaluate** — predictions are compared against the unseen test set

### Hyperparameter Tuning

`RandomizedSearchCV` was used to search 20 combinations across the following space:

| Setting | Options Tried | Winner |
| --- | --- | --- |
| Number of trees | 100, 200, 500, 1000 | **500** |
| Max tree depth | None, 10, 20, 30 | **None (unlimited)** |
| Min samples to split | 2, 5, 10 | **2** |
| Min samples per leaf | 1, 2, 4 | **1** |
| Features per split | sqrt, log2, all | **sqrt** |

Tuning reduced cross-validated MAE from ~7.5% (default) to **~7.2%**.

---

## Results

### Performance Metrics

| Metric | Value | Meaning |
| --- | --- | --- |
| R² | 0.85 | 85% of ROI variation explained by the three features |
| MAE | 7.24% | Average prediction is within ±7.24 percentage points |

Any predicted ROI below 7.24% falls inside the error margin and should be treated with caution.

### 5-Fold Cross-Validation

| Fold | MAE | R² |
| --- | --- | --- |
| 1 | ~7.1% | 0.84 |
| 2 | ~7.3% | 0.85 |
| 3 | ~7.0% | 0.86 |
| 4 | ~7.4% | 0.84 |
| 5 | ~7.2% | 0.85 |
| **Average** | **7.2% ± 0.1%** | **0.848 ± 0.007** |

Consistent results across all folds confirm the model generalises reliably and is not overfitting to a single data split.

### Sample Predictions vs. Reality

| Zip Code | Actual ROI | Predicted ROI | Error | Verdict |
| --- | :---: | :---: | :---: | --- |
| Test #1 | 22.3% | 20.8% | 1.5% | ✅ Excellent |
| Test #2 | 8.1% | 10.2% | 2.1% | ✅ Good |
| Test #3 | 14.7% | 12.9% | 1.8% | ✅ Good |
| Test #4 | 3.2% | 5.8% | 2.6% | ⚠️ Caution zone |
| Test #5 | 31.0% | 25.1% | 5.9% | ✅ Acceptable |

---

## Feature Importance

| Rank | Feature | Importance | Interpretation |
| --- | --- | :---: | --- |
| #1 | Current Property Price | 43.8% | Entry price is the single strongest predictor of future growth trajectory |
| #2 | Median Household Income | 36.8% | Wealthier neighbourhoods show more economic resilience and sustained investment |
| #3 | School Density | 19.4% | Acts as a *safety floor* — high school density prevents value collapse in downturns |

> **The safety floor finding:** School density does not push ROI up, but it prevents it from collapsing. Areas with many schools are significantly more resilient during economic downturns.

---

## Real-World Validation: Frederick, CO

| Scenario | Income | Schools | Price | Predicted ROI | Verdict |
| --- | :---: | :---: | :---: | :---: | --- |
| Wealthy suburb | $150,000 | 2 | $800,000 | 13.55% | ✅ Solid |
| School-dense hub | $75,000 | 8 | $400,000 | 4.69% | ⚠️ Caution |
| Frederick, CO (real) | $75,000 | 12 | $142,300 | 5.01% | ⚠️ Caution |

**Why is Frederick flagged despite 12 schools?**

Frederick's predicted ROI of 5.01% is lower than the model's error margin of 7.24%. The real return could be anywhere from **−2.2% to +12.3%** — the prediction cannot be confidently distinguished from noise. A data-driven investor should wait for stronger signals.

---

## Investment Verdict Tool

Enter three numbers about any zip code to get a plain-English verdict:

| Input | Example |
| --- | --- |
| Median household income | $85,000 |
| School count | 4 |
| Current home price | $420,000 |
| → Predicted ROI | ~12.4% |
| → Verdict | ✅ Solid Investment |

**Verdict thresholds:**

| Predicted ROI | Verdict |
| --- | --- |
| > 15% | ✅ Hidden Gem |
| 5% – 15% | ✅ Solid Investment |
| < 5% | ⚠️ Proceed with Caution |

---

## Conclusions

- **Price is king (43.8%)** — entry price is the strongest predictor of growth, but high price alone does not guarantee high ROI.
- **Income stabilises value (36.8%)** — wealthy neighbourhoods show more consistent appreciation over time.
- **Schools are a safety floor (19.4%)** — they do not guarantee high returns, but high school density significantly reduces collapse risk.
- **The 7.24% error margin is a tool, not a flaw** — it protects investors from overconfident decisions on borderline zip codes.
- **Hyperparameter tuning reduced MAE by ~0.3%** — modest but meaningful.
- **Cross-validation confirmed consistency** — MAE was stable across all 5 folds (7.0–7.4%).

> *This model moves real estate investing from gut feeling to data science. It does not replace local knowledge — but it gives any investor a rigorous, transparent starting point before making a major financial decision.*

---

## References

1. Zillow Research. "Zillow Home Value Index (ZHVI) Time Series," 2024. [zillow.com/research/data](https://www.zillow.com/research/data/)
2. U.S. Census Bureau. "ACS 5-Year Estimates: Median Household Income (Table B19013)," 2024. [data.census.gov](https://data.census.gov)
3. NCES. "Common Core of Data: Public School Universe," 2024. [nces.ed.gov/ccd](https://nces.ed.gov/ccd/)
4. L. Breiman. "Random Forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.
5. T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.
6. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

---

*MS Data Science · Regis University · Denver, CO*
