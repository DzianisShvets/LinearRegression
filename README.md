Step 1: Import Libraries
These libraries and modules are essential for the entire workflow:

pandas, numpy: For data manipulation.
matplotlib.pyplot, seaborn: For plotting.
sklearn.*: For data preprocessing, modeling, evaluation, and dataset loading.

Step 2: Load Dataset
data = fetch_openml(name='house_prices', as_frame=True)
df = data.frame
Loads the Ames housing dataset from OpenML, a real-world dataset with many features about house properties and the price (SalePrice).

It loads as a DataFrame for easier processing.

Step 3: Remove Outliers
q_low = df['SalePrice'].quantile(0.01)
q_high = df['SalePrice'].quantile(0.99)
df = df[(df['SalePrice'] > q_low) & (df['SalePrice'] < q_high)]
Outliers can distort predictions. Here, you keep only the middle 98% of data by removing the cheapest and most expensive 1%.

Step 4: Separate Features and Target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
X holds all input features (independent variables).

y holds the target (dependent variable), which is the sale price.

Step 5: Identify Feature Types
numeric_features = ...
categorical_features = ...
Splits features into:

Numerical: e.g., square footage, year built.

Categorical: e.g., type of roof, neighborhood.

Step 6: Create Pipelines
Numerical Pipeline:

Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
Handles missing values by filling with the mean.

Scales data so all features have the same scale (important for regression).

Categorical Pipeline:

Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
Replaces missing strings with the most common value.

Encodes strings into numbers using One-Hot Encoding.

Step 7: Combine Pipelines
ColumnTransformer([...])
Merges both pipelines into one transformer:

Applies numerical pipeline to numeric columns.

Applies categorical pipeline to categorical columns.

Step 8: Full Pipeline
Pipeline([
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])
Chains preprocessing and modeling in one flow:

First, data is cleaned and encoded.

Then, it's passed into the regression model.

Step 9: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(...)
80% of the data is used for training.

20% is held back for testing the model's performance.

Step 10: Train Model
model.fit(X_train, y_train)
Fits the model on preprocessed training data.

Internally, preprocessing steps are applied automatically.

Step 11: Predict
y_pred = model.predict(X_test)
Uses the trained model to predict house prices on the test data.

Step 12: Evaluate
r2_score, mean_absolute_error
R² Score: Measures how well the model explains variance in the target.

1.0 is perfect, 0.0 means no correlation.

MAE: Average difference between predicted and actual price (in $).

Step 13: Plot Actual vs Predicted
plt.scatter(y_test, y_pred, ...)
Visual check of how well predictions match actual values.

The red dashed line is the ideal line: predicted = actual.

Good models will show points close to this line.

Step 14: Feature Importance
get_feature_names_out, model.named_steps['regression'].coef_
