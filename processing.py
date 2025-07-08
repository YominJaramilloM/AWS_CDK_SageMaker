#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  
import os

input_path = "/opt/ml/processing/input/data.csv"

df = pd.read_csv(input_path)
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]
print(X.shape, y.shape)

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['ocean_proximity']
print("Numéricas:", num_features)
print("Categóricas:", cat_features)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Opcional para NaNs en la categórica
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

processing_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

X_train_processed = processing_pipe.fit_transform(X_train)
X_test_processed = processing_pipe.transform(X_test)

output_dir = "/opt/ml/processing/output"  #for S3

pd.DataFrame(X_train_processed).to_csv(f"{output_dir}/X_train.csv", index=False)
pd.DataFrame(X_test_processed).to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

print("Preprocessing completed and files saved.")

