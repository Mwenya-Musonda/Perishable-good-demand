# =====================================
# 🧠 PREDICTING DEMAND FOR PERISHABLE GOODS
# =====================================

# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# =====================================
# STEP 1: LOAD DATA
# =====================================
product_detail = pd.read_csv("/workspaces/Perishable-good-demand/Data/product_details.csv")
store_info = pd.read_csv("/workspaces/Perishable-good-demand/Data/store_info.csv")
supplier_info = pd.read_csv("/workspaces/Perishable-good-demand/Data/supplier_info.csv")
weather_data = pd.read_csv("/workspaces/Perishable-good-demand/Data/weather_data.csv")
weekly_sales = pd.read_csv("/workspaces/Perishable-good-demand/Data/weekly_sales.csv")

# =====================================
# STEP 2: EDA FUNCTION
# =====================================
def eda_stage_1(data, name):
    print("=" * 100)
    print(f"{name} dataset info")
    print(data.info())
    print("=" * 100)
    print(f"{name} dataset shape: {data.shape}")
    print("=" * 100)
    print(f"{name} dataset description")
    print(data.describe())
    print("=" * 100)
    print(f"{name} dataset missing values")
    print(data.isna().sum())
    print("=" * 100)
    print(f"{name} dataset duplicates: {data.duplicated().sum()}")

eda_stage_1(product_detail, "Product Detail")
eda_stage_1(supplier_info, "Supplier Info")
eda_stage_1(store_info, "Store Info")
eda_stage_1(weekly_sales, "Weekly Sales")
eda_stage_1(weather_data, "Weather Data")

# =====================================
# STEP 3: BASIC DESCRIPTIVE ANALYSIS
# =====================================
print(f"We have {len(product_detail['Product_Category'].value_counts())} product categories.")
print(f"We have {len(product_detail)} total products in store.")
print(f"The sales dataset covers {len(weekly_sales)} weeks.")
print(f"Total units sold: {weekly_sales['Units_Sold'].sum():,}")
print(f"Total wastage units: {weekly_sales['Wastage_Units'].sum():,}")
print(f"Average wastage units per region: {weekly_sales['Wastage_Units'].mean():,.2f}")

# =====================================
# STEP 4: VISUAL ANALYSIS
# =====================================
# --- Sales and Wastage by Region ---
plt.figure(figsize=(15, 12))

# 1. Stores per Region
plt.subplot(2, 2, 1)
region_store_counts = store_info['Region'].value_counts()
sns.barplot(x=region_store_counts.index, y=region_store_counts.values)
plt.title("Number of Stores by Region")

# 2. Average Store Size
plt.subplot(2, 2, 2)
store_size_by_region = store_info.groupby('Region')['Store_Size'].mean()
sns.barplot(x=store_size_by_region.index, y=store_size_by_region.values)
plt.title("Average Store Size by Region")

# 3. Total Units Sold
plt.subplot(2, 2, 3)
region_sales = weekly_sales.merge(store_info, on="Store_ID")
sales_region = region_sales.groupby('Region')['Units_Sold'].sum().sort_values(ascending=False)
sns.barplot(x=sales_region.index, y=sales_region.values)
plt.title("Total Units Sold per Region")

# 4. Total Wastage
plt.subplot(2, 2, 4)
wastage_region = region_sales.groupby('Region')['Wastage_Units'].sum().sort_values(ascending=False)
sns.barplot(x=wastage_region.index, y=wastage_region.values)
plt.title("Wastage per Region")

plt.tight_layout()
plt.show()

# =====================================
# STEP 5: MERGE ALL DATASETS
# =====================================
merged_data = (
    product_detail
    .merge(weekly_sales, on='Product_ID', how='inner')
    .merge(store_info, on='Store_ID', how='inner')
    .merge(supplier_info, on='Supplier_ID', how='inner')
    .merge(weather_data, on=['Week_Number', 'Region'], how='inner')
)

# =====================================
# STEP 6: FEATURE SELECTION (MUTUAL INFORMATION)
# =====================================
from sklearn.feature_selection import mutual_info_regression

target = merged_data['Units_Sold']
X = merged_data.drop(columns=['Units_Sold'])

# Remove ID columns
X.drop(columns=['Product_ID', 'Supplier_ID', 'Store_ID'], inplace=True)

# Encode categorical columns
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = pd.Categorical(X[col]).codes

# Compute MI scores
mi_scores = mutual_info_regression(X, target, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

print(feature_importance.head(10))

# Top 10 Features
training_features = feature_importance.head(10)['Feature'].tolist()
print("Top Predictors:", training_features)

# =====================================
# STEP 7: LINEAR REGRESSION MODEL
# =====================================
from sklearn.linear_model import LinearRegression

def train_linear_model(data, training_features):
    X = data[training_features].copy()
    y = data['Units_Sold']

    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Linear Regression Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape * 100:.2f}%")
    return model

model_lin = train_linear_model(merged_data, training_features)

# =====================================
# STEP 8: DECISION TREE REGRESSOR
# =====================================
from sklearn.tree import DecisionTreeRegressor

def train_decision_model(data, training_features):
    X = data[training_features].copy()
    y = data['Units_Sold']

    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Decision Tree Model Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape * 100:.2f}%")
    return model

model_tree = train_decision_model(merged_data, training_features)

# =====================================
# STEP 9: RANDOM FOREST REGRESSOR
# =====================================
from sklearn.ensemble import RandomForestRegressor

def train_rf_model(data, training_features):
    X = data[training_features].copy()
    y = data['Units_Sold']

    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Random Forest Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape * 100:.2f}%")
    return model

model_rf = train_rf_model(merged_data, training_features)

# =====================================
# STEP 10: XGBOOST REGRESSOR
# =====================================
from xgboost import XGBRegressor

def train_xg_model(data, training_features):
    X = data[training_features].copy()
    y = data['Units_Sold']

    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("XGBoost Model Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape * 100:.2f}%")

    return model

model_xgb = train_xg_model(merged_data, training_features)
