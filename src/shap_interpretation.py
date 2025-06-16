# shap_interpretation.py
import shap
import matplotlib.pyplot as plt


def explain_with_shap(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
    return shap_values