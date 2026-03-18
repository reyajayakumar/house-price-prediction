#imports pandas library and reads a CSV file named "housing_data.csv" into a DataFrame
import pandas as pd

# Goal: Predict housing prices using numerical features
df = pd.read_csv("housing_data.csv")
#prints the first few rows of the DataFrame to the console
print(df.head())

print(df.info()) #tells us about the data types of each column and if there are any missing values
print(df.describe()) #provides summary statistics for the numerical columns in the DataFrame, such as mean, standard deviation, minimum, and maximum values

print(df.isnull().sum().sort_values(ascending=False).head(10)) 
#shows the number of missing values in each column, sorted in descending order

import matplotlib.pyplot as plt
import seaborn as sns
#creates a histogram of the "price" column in the DataFrame

sns.histplot(df["price"], bins=30)
plt.title("House Price Distribution")
plt.show()

corr = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
print(corr.head(10))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# select features (input variables)
features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
features.remove("price")  # target column

X = df[features]
y = df["price"]

# handle missing values (simple way)
X = X.fillna(X.mean())

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.3f}")

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print(f"Random Forest R^2 Score: {rf_score:.3f}")

print("Top features affecting price:")
print(corr.head(5))