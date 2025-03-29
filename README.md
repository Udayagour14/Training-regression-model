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
```
### 3️⃣ Choose Regression Model
`✅ Select a regression algorithm based on data characteristics:

Linear Regression (if data is linear).

Polynomial Regression (for non-linear trends).

Decision Tree/Random Forest (for complex relationships).

Lasso/Ridge Regression (for regularization)`.
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```
### 4️⃣ Train the Model
`✅ Fit the model to training data.
model.fit(X_train, y_train)`


