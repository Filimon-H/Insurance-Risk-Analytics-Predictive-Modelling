import mlflow
import mlflow.sklearn
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def initialize_mlflow(uri="notebook", experiment_name="Default"):
    """
    Initializes MLflow experiment and returns the experiment ID.
    """
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id

def train_and_log_model(model, model_name, tracking_id, x_train, y_train, x_test, y_test):
    """
    Train the model, evaluate it, and log to MLflow.
    Returns: MSE, R2, trained_model
    """
    with mlflow.start_run(experiment_id=tracking_id, run_name=model_name):
        # Train model
        model.fit(x_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

    return mse, r2, model

def explain_model_with_shap(model, x_train, x_test, max_display=15):
    """
    Use SHAP to explain the trained model and show summary plot.
    Returns: SHAP values for the test set.
    """
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_test)

    # Plot SHAP summary
    shap.summary_plot(shap_values, x_test, max_display=max_display)

    return shap_values
