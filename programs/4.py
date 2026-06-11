import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
n = 200

glucose    = np.random.randint(80, 200, n)
bmi        = np.random.uniform(18, 45, n)
gender     = np.random.randint(0, 2, n)           # 0=female, 1=male
blood      = np.random.randint(60, 120, n)
pregnancy  = np.random.randint(0, 10, n)
insulin    = np.random.randint(50, 250, n)

diabetes = ((( glucose > 130) & (bmi > 30)) | (insulin > 180)).astype(int)

df = pd.DataFrame({
    'Glucose': glucose, 'BMI': bmi, 'Gender': gender,
    'BloodPressure': blood, 'Pregnancies': pregnancy,
    'Insulin': insulin, 'Diabetes': diabetes
})

print("=== Generated Dataset (first 10 rows) ===")
print(df.head(10))

X = df.drop('Diabetes', axis=1).values
y = df['Diabetes'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== Normalized Data (first 5 rows) ===")
print(pd.DataFrame(X_scaled, columns=df.columns[:-1]).head())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
model.fit(X_train, y_train)

print("\n=== Training Loss per Iteration (last 10) ===")
print(model.loss_curve_[-10:])

plt.plot(model.loss_curve_)
plt.title("Cost Function")
plt.xlabel("Iterations"); plt.ylabel("Loss")
plt.show()

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n=== Accuracy ===")
print(f"Accuracy: {acc * 100:.2f}%")

print("\n=== Actual vs Predicted (first 15) ===")
print("Actual:   ", y_test[:15])
print("Predicted:", y_pred[:15])

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))