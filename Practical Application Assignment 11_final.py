#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = "/Users/krista.rime/Downloads/practical_application_II_starter/data/vehicles.csv"
data = pd.read_csv(file_path)


# In[2]:


# Filtering out extreme outliers in 'price' for demonstration purposes
data = data[(data['price'] > 1000) & (data['price'] < 100000)]


# In[3]:


# Selecting features to include in the model
features = ['year', 'odometer', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']
X = data[features]
y = data['price']


# In[4]:


# Handling missing values and encoding categorical variables
numeric_features = ['year', 'odometer']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[5]:


# Creating a modeling pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])


# In[6]:


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Fitting the model
model.fit(X_train, y_train)


# In[8]:


# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[9]:


# Evaluating the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)


# In[10]:


print(f'Training RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')


# In[41]:


# Plotting actual vs. predicted prices
plt.figure(figsize=(12, 6))

# Scatter plot for training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--', lw=2)
plt.title('Actual vs. Predicted Prices (Training Set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

# Scatter plot for test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.title('Actual vs. Predicted Prices (Test Set)')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.tight_layout()
plt.show()

# Plotting residuals
plt.figure(figsize=(10, 6))
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

# Residuals for training set
plt.subplot(1, 2, 1)
sns.histplot(residuals_train, bins=50, kde=True, color='blue')
plt.title('Residuals Distribution (Training Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Residuals for test set
plt.subplot(1, 2, 2)
sns.histplot(residuals_test, bins=50, kde=True, color='green')
plt.title('Residuals Distribution (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[45]:


# Extracting coefficients from the trained Linear Regression model
coefficients = model.named_steps['regressor'].coef_
feature_names = numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(input_features=categorical_features))

# Create a DataFrame to store feature coefficients
feature_coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Function to aggregate coefficients by the target variable
def aggregate_coefficients_by_target(df, target_column):
    aggregated_df = df.copy()
    # Extracting the target variable from the feature names
    aggregated_df[target_column] = aggregated_df['Feature'].str.split('_').str[0]
    # Grouping by the target variable and calculating the average coefficient
    aggregated_df = aggregated_df.groupby(target_column)['Coefficient'].mean().reset_index()
    return aggregated_df

# Aggregating coefficients by the target variable (price)
aggregated_coefficients_price = aggregate_coefficients_by_target(feature_coefficients_df, 'price')

# Printing aggregated coefficients by price
print("Aggregated Coefficients by Price:")
print(aggregated_coefficients_price)


# In[46]:


# Plotting aggregated coefficients by price
plt.figure(figsize=(10, 6))
plt.barh(aggregated_coefficients_price['price'], aggregated_coefficients_price['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.ylabel('Price')
plt.title('Aggregated Coefficients by Price')
plt.grid(axis='x')
plt.show()


# In[47]:


# Remove 'year' and 'odometer' from the aggregated coefficients dataframe
aggregated_coefficients_price = aggregated_coefficients_price[aggregated_coefficients_price['price'] != 'year']
aggregated_coefficients_price = aggregated_coefficients_price[aggregated_coefficients_price['price'] != 'odometer']

# Plotting aggregated coefficients by price
plt.figure(figsize=(10, 6))
plt.barh(aggregated_coefficients_price['price'], aggregated_coefficients_price['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.ylabel('Price')
plt.title('Aggregated Coefficients by Price')
plt.grid(axis='x')
plt.show()


# In[ ]:




