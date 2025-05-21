# disease_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Combine all symptom columns into one set to find all unique symptoms
symptom_columns = [col for col in df.columns if 'Symptom' in col]
all_symptoms = set()

for col in symptom_columns:
    df[col] = df[col].str.strip().str.lower().fillna('')
    all_symptoms.update(df[col].unique())

# Remove empty strings
all_symptoms.discard('')

# Create binary features for each symptom
for symptom in all_symptoms:
    df[symptom] = df[symptom_columns].apply(lambda x: symptom in x.values, axis=1).astype(int)

# Features and target
X = df[list(all_symptoms)]
y = df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and symptom list
with open('disease_model.pkl', 'wb') as f:
    pickle.dump((model, list(all_symptoms)), f)
