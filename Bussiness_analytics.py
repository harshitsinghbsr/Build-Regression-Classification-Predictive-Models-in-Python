import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# read the CSV file into a pandas DataFrame
Employes = pd.read_csv("https://github.com/ybifoundation/Dataset/raw/main/EmployeeAttrition.csv")

# split the data into training and testing sets
X = Employes[['Age','DailyRate','Education','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction',
             'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome',
              'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
               'StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
                'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager' ]]
y = Employes ['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit a KNN model to the training data
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

# make predictions on the test data
y_pred = classifier.predict(X_test)

# evaluate the model using a classification report
print(classification_report(y_test, y_pred))
