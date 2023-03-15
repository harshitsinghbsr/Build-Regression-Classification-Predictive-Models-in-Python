import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# read the CSV file into a pandas DataFrame
cement = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv")

cement.head()
cement.info()
cement.describe()

# split the data into training and testing sets
X = cement[['Cement (kg in a m^3 mixture)','Blast Furnace Slag (kg in a m^3 mixture)',
    'Fly Ash (kg in a m^3 mixture)','Water (kg in a m^3 mixture)','Superplasticizer (kg in a m^3 mixture)',
    'Coarse Aggregate (kg in a m^3 mixture)','Fine Aggregate (kg in a m^3 mixture)','Age (day)']]
y = cement ['Concrete Compressive Strength(MPa, megapascals) ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit a linear regression model to the training data
regressor = GradientBoostingRegressor(n_estimators=130, learning_rate=0.2, max_depth=6, random_state=2523)
regressor.fit(X_train, y_train)

# make predictions on the test data
y_pred = regressor.predict(X_test)

# evaluate the model using mean squared error
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)