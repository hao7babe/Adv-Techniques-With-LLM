import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'income_ds.csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop('High_Income', axis=1)
y = data['High_Income']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing pipeline for both numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Split the data into training (75%) and testing (25%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model 1: Feed Forward Neural Network with 1 hidden layer (6 units)
mlp_1layer = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(6,), max_iter=500, random_state=42))
])

# Train the model
mlp_1layer.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_1layer = mlp_1layer.predict(X_test)
accuracy_1layer = accuracy_score(y_test, y_pred_1layer)

# Model 2: Feed Forward Neural Network with 2 hidden layers (8 and 4 units)
mlp_2layers = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=42))
])

# Train the model
mlp_2layers.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_2layers = mlp_2layers.predict(X_test)
accuracy_2layers = accuracy_score(y_test, y_pred_2layers)

# Print the results
print(f"Accuracy of model with 1 hidden layer (6 units): {accuracy_1layer:.3f}")
print(f"Accuracy of model with 2 hidden layers (8 and 4 units): {accuracy_2layers:.3f}")
