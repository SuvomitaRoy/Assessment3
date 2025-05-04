import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("train.csv")
df = pd.DataFrame(data)
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Fare'],axis='columns',inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())
enc = OneHotEncoder(sparse_output=False)
sex_enc = enc.fit_transform(np.array(df["Sex"]).reshape(-1,1))
one_hot_df = pd.DataFrame(sex_enc,columns=['Female','Male'])
df_new = pd.concat([df, one_hot_df], axis=1)
df_new = df_new.drop("Sex", axis=1)
scaler = StandardScaler()
scaler.fit_transform(df_new)

X = df_new.drop("Survived",axis=1,inplace=False)
y = df_new["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train,y_train)


# Load the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    prediction = model.predict(X_test)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
