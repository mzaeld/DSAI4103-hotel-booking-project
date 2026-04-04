# DSAI4103-hotel-booking-project
This project uses machine learning to predict hotel booking cancellations and provide actionable insights.

## Objective
To build a predictive model that identifies high-risk bookings and supports better decision-making.

## Models Used
- Logistic Regression
- Random Forest
- XGBoost (Final Model)

## Results
- Accuracy: ~84%
- Balanced precision and recall
- XGBoost selected as final model

## Explainability
SHAP was used to interpret model predictions and identify key drivers.

## Bias Analysis
Model performance was evaluated across customer segments with no significant bias detected.

## Dashboard
Power BI dashboard provides interactive visualization of cancellation risk.

## Files
- `final_project.ipynb` – main notebook
- `scoring.py` – scoring function
- `xgboost_model.pkl` – trained model
- `preprocessor.pkl` – preprocessing pipeline
- `scored_data.csv` – prediction output
- `hotel_dashboard.pbix` – Power BI dashboard