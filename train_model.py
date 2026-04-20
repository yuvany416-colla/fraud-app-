import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# sample dataset (you can replace later with real data)
data = {
    "distance_from_home": [1, 2, 3, 4, 5],
    "distance_from_last_transaction": [1, 1, 2, 2, 3],
    "ratio_to_median_purchase_price": [1, 2, 1, 2, 3],
    "repeat_retailer": [0, 1, 0, 1, 0],
    "used_chip": [1, 1, 0, 0, 1],
    "used_pin_number": [0, 1, 0, 1, 0],
    "online_order": [1, 0, 1, 0, 1],
    "fraud": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("fraud", axis=1)
y = df["fraud"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "logistic_regression_model.pkl")

print("Model saved successfully!")