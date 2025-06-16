# model_building.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def build_linear_model():
    return LinearRegression()


def build_random_forest(n_estimators=100, max_depth=None):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


def build_xgboost_model():
    return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)