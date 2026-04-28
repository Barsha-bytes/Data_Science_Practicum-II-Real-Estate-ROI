# Multi-Factor ROI Optimization: Colorado Real Estate Dynamics

**Author:** Barsha Kakshapati  
**Institution:** Regis University | M.S. Data Science  
**Status:** Master's Practicum II (Final Project)

## 📌 Project Overview
This practicum develops a machine learning framework to predict 3-year Return on Investment (ROI) for residential real estate in Colorado. By synthesizing longitudinal data from Zillow, the U.S. Census Bureau, and the National Center for Education Statistics (NCES), the study moves beyond historical pricing to identify "safety floors"—community infrastructure metrics that stabilize property values during market volatility.

## 🚀 Key Features
* **Random Forest Regressor:** An ensemble approach capturing non-linear relationships with an $R^2$ of 0.85.
* **Feature Engineering:** Integration of School Density and Median Household Income at the ZCTA (Zip Code) level.
* **Risk Management Framework:** Implementation of a 7.24% Mean Absolute Error (MAE) threshold to filter speculative "noise" from actionable investment signals.

## 📊 Visual Results & Analysis

### 1. Market Driver Correlation
The heatmap reveals that while neighborhood wealth (Median Income) strongly dictates the price of entry, it does not act as a primary predictor for future ROI.
![Correlation Heatmap](images/heatmap.png)

### 2. The "Safety Floor" Hypothesis
Feature importance analysis shows that school density is a critical secondary driver (19.4%), providing a demand baseline that protects property value.
![Feature Importance](images/Clean_feature_importance.png)

### 3. Model Validation (Frederick, CO)
Real-world validation shows a predicted ROI of 5.01%. Because this falls below our MAE threshold of 7.24%, the model issues a **Caution** verdict, prioritizing capital preservation.
![Frederick Stress Test](images/Frederick_stress_test.png)

## 📂 Repository Structure
* `/data`: Preprocessed datasets and data dictionary.
* `/notebooks`: Jupyter notebooks detailing EDA, Feature Engineering, and Model Tuning.
* `/src`: Python scripts for the automated data pipeline.
* `/images`: Visualizations and plots used in the report.
* `Practicum_Report.pdf`: Final academic documentation for Regis University.
* `IEEEtran.cls`: LaTeX class file for professional academic formatting.

## 🛠️ Data Dictionary
| Feature | Source | Description |
| :--- | :--- | :--- |
| **ZHVI** | Zillow | Zillow Home Value Index (Target Variable) |
| **Income_Median** | U.S. Census | Median household income per Zip Code |
| **School_Count** | NCES | Total number of public/charter schools within ZCTA |
| **ROI_3YR** | Calculated | The percentage change in property value over 36 months |

## ⚙️ Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Barsha-bytes/Data_Science_Practicum-II-Real-Estate-ROI.git](https://github.com/Barsha-bytes/Data_Science_Practicum-II-Real-Estate-ROI.git)
