import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna.visualization as vis
import numpy as np
import pandas as pd
from Podcast import eda

dataset = "Podcast/datasets/train.csv"
config_path = "../preprocessing_config.yaml"
target_col = 'Listening_Time_minutes'

data = pd.read_csv(dataset)
target = data[target_col]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "rmse",
    }

    model = xgb.XGBRegressor(**params)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    n_init = 0

    for train_idx, val_idx in kf.split(data):
        print(f"{n_init} KFold Started: ")
        builder = eda.PreprocessingPipelineFromConfig(config_path)
        pipeline = builder.load_config()

        x_train, x_val = data.iloc[train_idx], data.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        # Train
        x_train = pipeline.fit(x_train).transform(x_train)
        x_train = x_train.drop(columns=['id', 'Listening_Time_minutes'], errors='ignore')

        # Validation
        x_val = x_val.drop(columns=['id', 'Listening_Time_minutes'], errors='ignore')
        x_val = pipeline.transform(x_val)

        y_train = y_train.loc[x_train.index]
        y_val = y_val.loc[x_val.index]

        model.fit(x_train, y_train, verbose=False)
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        n_init += 1

    return np.mean(rmse_list)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best hyperparameters:", best_params)

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()