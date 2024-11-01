# Tips_Prediction_Project
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('../data/tips.csv')

# Step 1: Understanding the Dataset
print(data.head())  # Display first few rows
print(data.info())  # Dataset summary

# Data Cleaning: Checking for missing values
print(data.isnull().sum())
data.dropna(inplace=True)  # Drop rows with missing values

# Step 2: Data Visualization
# Scatter plot of total bill vs tip
plt.figure(figsize=(10, 6))
plt.scatter(data['total_bill'], data['tip'], color='blue')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.grid(True)
plt.show()

# Step 3: Model Building
# Defining the independent (X) and dependent (y) variables
X = data[['total_bill']]
y = data['tip']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Comparison of Actual vs Predicted Values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Tips')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.legend()
plt.grid(True)
plt.show()
