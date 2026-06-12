import pandas as pd   # for reading and working with data tables
from sklearn.ensemble import RandomForestClassifier  # our prediction model

#Initially used these two lines of code to read the data from my folder but had to change it so that it can be viewed on other systems as well. Had to use AI to make this change
#train = pd.read_csv("/Users/rohaan/train.csv")
#test  = pd.read_csv("/Users/rohaan/test.csv") 
import os
base = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(base, "train.csv"))
test  = pd.read_csv(os.path.join(base, "test.csv"))



print("Training data shape:", train.shape)   
print("Test data shape    :", test.shape)
print()



train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"]  = test["Age"].fillna(test["Age"].median())


test["Fare"] = test["Fare"].fillna(test["Fare"].median())


train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Embarked"]  = test["Embarked"].fillna(test["Embarked"].mode()[0])




train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"]  = test["Sex"].map({"male": 0, "female": 1})


train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"]  = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X_train = train[features]   # input features for training
y_train = train["Survived"] # correct answers for training
X_test  = test[features]    # input features we want predictions for



model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # "teach" the model using training data

print("Model training done!")
print()


predictions = model.predict(X_test)  # returns 0 or 1 for each passenger


output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

##Initially used this line of code to read the data from my folder but had to change it so that it can be viewed on other systems as well. Had to use AI to make this change
#output.to_csv("/Users/rohaan/submission.csv", index=False)
output.to_csv(os.path.join(base, "submission.csv"), index=False)

print("Predictions saved to submission.csv!")
print()
print("Preview of first 10 predictions:")
print(output.head(10))