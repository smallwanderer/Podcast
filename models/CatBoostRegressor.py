import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna.visualization as vis
import numpy as np
import pandas as pd
from Podcast import eda

dataset = "../datasets/train.csv"
config_path = "../preprocessing_config.yaml"
target_col = 'Listening_Time_minutes'

data = pd.read_csv(dataset)
target = data[target_col]

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_seed": 42,
        "verbose": 0
    }

    model = CatBoostRegressor(**params)

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

        model.fit(x_train, y_train)
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