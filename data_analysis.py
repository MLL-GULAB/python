import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Data Collection and Cleaning
# Create sample data
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'sales': np.random.randint(50, 200, size=100),
    'marketing_spend': np.random.randint(1000, 5000, size=100),
    'number_of_customers': np.random.randint(10, 100, size=100)
}

df = pd.DataFrame(data)

# Step 2: Data Exploration
# Descriptive statistics
print(df.describe())

# Visualization
# Histogram of sales
sns.histplot(df['sales'])
plt.title('Distribution of Sales')
plt.show()

# Scatter plot of marketing spend vs. sales
sns.scatterplot(x='marketing_spend', y='sales', data=df)
plt.title('Marketing Spend vs. Sales')
plt.show()

# Step 3: Data Analysis
# Correlation analysis
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Trend analysis using time series
df.set_index('date', inplace=True)
df['sales'].plot()
plt.title('Sales Over Time')
plt.show()

# Step 4: Predictive Modeling
# Prepare data
X = df[['marketing_spend', 'number_of_customers']]
y = df['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 5: Data Visualization and Reporting
# Plot predictions vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
