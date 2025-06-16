from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
    }

def train_and_evaluate(models, x_train, y_train, x_test, y_test, train_func, tracking_id):
    best_model = None
    best_r2 = -float('inf')
    best_mse = float('inf')

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        mse, r2, trained_model = train_func(model, model_name, tracking_id, x_train, y_train, x_test, y_test)
        
        if r2 > best_r2:
            best_r2 = r2
            best_mse = mse
            best_model = trained_model

    print(f"Best model: {best_model.__class__.__name__}")
    print(f"Best R2 Score: {best_r2}")
    print(f"Best MSE: {best_mse}")

    return best_model, best_r2, best_mse
