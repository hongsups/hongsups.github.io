from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
print("Model trained and saved as 'model.joblib'")
