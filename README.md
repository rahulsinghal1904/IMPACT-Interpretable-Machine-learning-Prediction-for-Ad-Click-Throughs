# IMPACT-Interpretable-Machine-learning-Prediction-for-Ad-Click-Throughs


Click-through rate (CTR) prediction is crucial in digital advertising to optimize
user engagement and maximize revenue. While advanced models like Wide &
Deep and LightGBM have achieved high accuracy, their lack of interpretability
limits real-world applicability, particularly in contexts requiring transparency and
accountability. To address this gap, we present IMPACT, a framework that balances
interpretability and predictive performance for CTR prediction. Our approach
incorporates exploratory data analysis (EDA) to uncover critical feature interactions,
advanced feature engineering to enhance contextual and temporal representations,
and rigorous hyperparameter tuning to optimize models such as LightGBM and
XGBoost. Using SHAP (SHapley Additive exPlanations), we provide actionable
insights into feature importance, ensuring model transparency. Experiments on the
Avazu dataset demonstrate that our method achieves a log-loss of 0.3915 on the test
set, outperforming existing baselines while delivering detailed interpretability. This
work paves the way for trust and scalability in real-world CTR prediction, ensuring
broader applicability across high-stakes domains like finance and healthcare.
