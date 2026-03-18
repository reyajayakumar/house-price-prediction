# House Price Prediction Project

# Project Overview
This project predicts house prices using a Kaggle dataset.  
It demonstrates data cleaning, exploratory data analysis (EDA), visualization, and predictive modeling using Python.

# Dataset
- Source: Kaggle – “House Prices - Advanced Regression Techniques”  
- File: `train.csv`  
- Description: Contains house features (size, number of rooms, year built, etc.) and the target variable `SalePrice`.

# Tools & Libraries
- Python 3.11 
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- IDE: Visual Studio Code  

# Project Steps
- Load the Data
- Imported the CSV file into a Pandas DataFrame  
- Explored the first few rows with `df.head()`  
- Checked column types and missing values with `df.info()` and `df.isnull().sum()`
- Stripped extra spaces from column names  
- Identified missing values for further cleaning in future iterations
- Plotted distribution of house prices with Seaborn  
- Checked correlation of features with `price` to identify key predictors

