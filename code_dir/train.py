#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import joblib
import os
import tarfile
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

input_dir = "/opt/ml/processing/input"
model_dir = "/opt/ml/model"

X_train = pd.read_csv(f"{input_dir}/X_train.csv")
X_test = pd.read_csv(f"{input_dir}/X_test.csv")
y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{input_dir}/y_test.csv").squeeze()

print(y_train.shape, y_test.shape)

param_grid = {
    'n_estimators': randint(20, 100),
    'max_depth': randint(5, 12),
    'min_samples_split': randint(2, 50)
}

model= RandomForestRegressor(random_state=42, n_jobs=-1)
randomS = RandomizedSearchCV(model, param_grid, n_iter=50,cv=10, scoring='r2', n_jobs=-1, random_state=42)
randomS.fit(X_train, y_train)
print("Best parameters:")
print(randomS.best_params_)

best_model = randomS.best_estimator_
y_pred_train = best_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

y_pred_test = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# save model as .joblib
model_path = os.path.join(model_dir, "model.joblib")
joblib.dump(best_model, model_path)

assert os.path.exists(model_path), "Model file not found before archiving!"

#  .tar.gz
with tarfile.open(os.path.join(model_dir, "model.tar.gz"), mode="w:gz") as archive:
    archive.add(model_path, arcname="model.joblib")

print("Model packaged correctly.")

