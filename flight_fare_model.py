
import pandas as pd
import numpy as np
import joblib  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# LOAD DATA
df = pd.read_csv("Clean_Dataset.csv")
print(df.shape)
print(df.dtypes)
print(df["price"].describe())

# DATA CLEANING
print("\nMissing Values:")
print(df.isnull().sum())

df.drop(columns=["Unnamed: 0", "flight"], errors="ignore", inplace=True)
df = df[df['price'] > 100] 
df.dropna(inplace=True)

# FEATURE ENGINEERING
df["stops"] = df["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})
df['is_urgent'] = (df['days_left'] <= 2).astype(int)

# DATA SPLITTING
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PREPROCESSING
num_features = ["duration", "days_left", "stops", "is_urgent"]
cat_features = ["airline", "source_city", "destination_city", "departure_time", "arrival_time", "class"]

preprocessor = ColumnTransformer([
    ("num", PowerTransformer(method='yeo-johnson'), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
])

# MODEL SELECTION
rf = RandomForestRegressor(
    n_estimators=100, 
    max_depth=20, 
    random_state=42, 
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", rf)
])

# HYPERPARAMETER TUNING
param_grid = {
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", None]
}

print("Optimizing model...")
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2')
grid.fit(X_train.sample(frac=0.2, random_state=42), y_train.sample(frac=0.2, random_state=42))

# FINAL TRAINING
best_model = grid.best_estimator_
print("Training final model...")
best_model.fit(X_train, y_train)

# EVALUATION
y_pred = best_model.predict(X_test)
print(f"\nR2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} ")

# SAVE  
joblib.dump(best_model, "flight_fare_model_fast.joblib", compress=3)
print("\nModel saved as flight_fare_model_fast.joblib")