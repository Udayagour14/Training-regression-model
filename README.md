## After encoding categorical variables (like One-Hot Encoding or Label Encoding), the following steps are followed
### 1️⃣ Data Preprocessing
✅ Handle Missing Values – Fill or remove missing data.
✅ Handle Outliers – Remove or transform extreme values.
✅ Normalize/Standardize Data (if required) – Scale numerical features to ensure equal importance.
### 2️⃣ Splitting Data
✅ Divide the dataset into training & testing sets (e.g., 80% training, 20% testing).
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

_________

```output
| R&D Spend  | Administration | Marketing Spend | State_Florida | State_New York | Profit     |
|------------|---------------|----------------|---------------|---------------|------------|
| 165349.2   | 136897.8      | 471784.1       | 0             | 1             | 192261.83  |
| 162597.7   | 151377.59     | 443898.53      | 0             | 0             | 191792.06  |
| 153441.51  | 101145.55     | 407934.54      | 1             | 0             | 191050.39  |
| 144372.41  | 118671.85     | 383199.62      | 0             | 1             | 182901.99  |
| 142107.34  | 91391.77      | 366168.42      | 1             | 0             | 166187.94  |
```
### Define Features (X) & Target (y)
### Now, we separate independent features (X) and dependent variable (y = Profit).
# Define independent variables (X) and dependent variable (y)
```python
X = df_encoded.drop(columns=['Profit'])  # Features
y = df_encoded['Profit']  # Target variable
```
### Train-Test Split (80-20%)
### We split the data into 80% training and 20% testing.
# Import train-test split function
``` python
from sklearn.model_selection import train_test_split  

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset sizes
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```
### 3️⃣ Choose Regression Model
`✅ Select a regression algorithm based on data characteristics:

Linear Regression (if data is linear).

Polynomial Regression (for non-linear trends).

Decision Tree/Random Forest (for complex relationships).

Lasso/Ridge Regression (for regularization)`.

### 4️⃣ Train the Model
✅ Fit the model to training data.
#### `model.fit(X_train, y_train)`

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
Train Multiple Linear Regression Model
Now, we train the model using Scikit-Learn.
```
```python
# Import Linear Regression
from sklearn.linear_model import LinearRegression  

# Initialize the model
lr_model = LinearRegression()  

# Train the model on training data
lr_model.fit(X_train, y_train)  

# Predict on both training & testing sets
y_pred_train = lr_model.predict(X_train)  
y_pred_test = lr_model.predict(X_test)
```

```python
# Import necessary metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error  

# Calculate Mean of Profit
mean_profit = y_test.mean()

# Calculate Errors
mae = mean_absolute_error(y_test, y_pred_test)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred_test)  # Mean Squared Error
rmse = mean_squared_error(y_test, y_pred_test, squared=False)  # Root Mean Squared Error

# Display results
mean_profit, mae, mse, rmse
```
```output
(112012.64, 7514.32, 81930549.19, 9055.96)
```
```✅ Interpretation:
Mean Profit = $112,012.64 → Average company profit.
MAE = $7,514.32 → On average, our predictions are $7,514 off.
MSE = 81,930,549.19 → Squared error, sensitive to outliers.
RMSE = $9,055.96 → On average, the error is around $9,056.
```
###  -------------------OR-----------------
### Model Evaluation (R² Score & RMSE)
We evaluate model accuracy using:
`R² Score → How well the model explains variation in Profit.
Root Mean Squared Error (RMSE) → Error between predicted and actual profit.`
```python
# Import metrics
from sklearn.metrics import r2_score, mean_squared_error  

# Compute R² Score & RMSE
r2_train = r2_score(y_train, y_pred_train)  
r2_test = r2_score(y_test, y_pred_test)  
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)  
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

# Display results
r2_train, r2_test, rmse_train, rmse_test
```
```(0.954, 0.899, 8927.49, 9055.96)```
```
✅ Interpretation:
R² (Train) = 0.954 → Model explains 95.4% of profit variance.
R² (Test) = 0.899 → Model explains 89.9% on unseen data.
RMSE (Train) = 8927.49, RMSE (Test) = 9055.96 → Model has small prediction error.
Slight Overfitting → Training accuracy is slightly higher than testing.
```






