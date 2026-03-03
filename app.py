import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load built-in sklearn heart-like dataset
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data["target"] = dataset.target

# Use only 3 features for simplicity
X = data.iloc[:, :3]
y = data["target"]

# Split data (important for hackathon credibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)