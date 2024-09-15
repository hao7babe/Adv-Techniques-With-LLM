import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('heart_attack.csv')

# Prepare the data
X = data.drop(columns=['heart_attack'])
y = data['heart_attack']

# Split the data into training and test sets (75% training, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train KNN with k=3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)

# Train KNN with k=5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)

# Predict with both models
y_pred_knn3 = knn3.predict(X_test)
y_pred_knn5 = knn5.predict(X_test)

# Calculate accuracy for both models
accuracy_knn3 = accuracy_score(y_test, y_pred_knn3)
accuracy_knn5 = accuracy_score(y_test, y_pred_knn5)

# Print accuracies
print(f"Accuracy of KNN with k=3: {accuracy_knn3}")
print(f"Accuracy of KNN with k=5: {accuracy_knn5}")
