import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

folder = os.path.dirname(os.path.abspath(__file__))

training_data = pd.read_csv(os.path.join(folder, "train.csv"))
testing_data  = pd.read_csv(os.path.join(folder, "test.csv"))

print(f"Rows in training set : {len(training_data)}")
print(f"Rows in testing set  : {len(testing_data)}")
print()

for dataset in [training_data, testing_data]:
    dataset["Age"]      = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])

testing_data["Fare"] = testing_data["Fare"].fillna(testing_data["Fare"].mean())

for dataset in [training_data, testing_data]:
    dataset["Sex"]      = dataset["Sex"].map({"male": 0, "female": 1})
    dataset["Embarked"] = dataset["Embarked"].map({"S": 0, "C": 1, "Q": 2})

selected_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

inputs        = training_data[selected_columns]
correct_label = training_data["Survived"]
to_predict    = testing_data[selected_columns]

model = LogisticRegression(max_iter=200)
model.fit(inputs, correct_label)

print("Training complete!")
print()

results = model.predict(to_predict)

final_output = pd.DataFrame({
    "PassengerId": testing_data["PassengerId"],
    "Survived"   : results
})

save_path = os.path.join(folder, "submission.csv")
final_output.to_csv(save_path, index=False)

print(f"submission.csv saved to: {save_path}")
print()
print("First 10 predictions:")
print(final_output.head(10))
