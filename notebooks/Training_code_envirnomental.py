import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
## load the dataset
df=pd.read_csv("crop_disease_data.csv")

### pre-processing
df=df.drop(columns=["Crop","Label"],axis=1)
df.isnull().sum()
df.info()
df.dropna()

for x in df.columns:
    print(df[x].value_counts())

# Split the data into features (X) and target (y)
X = df.iloc[:,0:4]  # Features (One-Hot Encoded)
y = df.iloc[:,4]  # Target (Label Encoded)

print(X.shape)

print(df["Disease"].value_counts())

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Save the trained model to a file
joblib.dump(rf_model, 'random_forest_model.pkl')

print("Model saved successfully!")

