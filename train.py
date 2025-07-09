# Step 1. Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Step 2. Load the dataset
data = fetch_openml(name='house_prices', as_frame=True)
df = data.frame

# Step 3. Remove extreme outliers (too cheap or too expensive houses)
# We use 1st and 99th percentiles to keep most of the data
q_low = df['SalePrice'].quantile(0.01)
q_high = df['SalePrice'].quantile(0.99)
df = df[(df['SalePrice'] > q_low) & (df['SalePrice'] < q_high)]

# Step 4. Separate features and target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Step 5. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Step 6. Create transformers for numeric and categorical data
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),         # Fill missing numbers
    ('scaler', StandardScaler())                         # Scale features
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing strings
    ('encoder', OneHotEncoder(handle_unknown='ignore'))    # Encode strings into 0/1
])

# Step 7. Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Step 8. Build the full pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])

# Step 9. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 10. Fit the model
model.fit(X_train, y_train)

# Step 11. Make predictions
y_pred = model.predict(X_test)

# Step 12. Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: ${mae:,.0f}")

# Step 13. Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 14. Show Feature Importance (Top 20)
# Extract preprocessed feature names
cat_encoded = model.named_steps['preprocessing']\
    .named_transformers_['cat'].named_steps['encoder']\
    .get_feature_names_out(categorical_features)

all_feature_names = np.concatenate([numeric_features, cat_encoded])

# Get regression coefficients
coefficients = model.named_steps['regression'].coef_

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
})

# Sort by absolute value of coefficients
feature_importance['AbsCoefficient'] = feature_importance['Coefficient'].abs()
top_features = feature_importance.sort_values(by='AbsCoefficient', ascending=False).head(20)

# Plot top 20 important features
plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='AbsCoefficient', y='Feature')
plt.title("Top 20 Important Features (by absolute coefficient)")
plt.xlabel("Importance (|Coefficient|)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()