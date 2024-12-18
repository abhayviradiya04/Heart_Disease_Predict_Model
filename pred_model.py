import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the data from heart.csv
df = pd.read_csv('heart.csv')

# Encode categorical variables if necessary (assuming columns like 'Sex', 'ChestPainType' etc. are in the dataset)
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})  # Example of mapping for 'Sex' (adjust if needed)
df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})  # Example for ChestPainType
df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# Define features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the RandomForest model with default parameters
model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model, features, and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')
joblib.dump(scaler, 'scaler.pkl')
