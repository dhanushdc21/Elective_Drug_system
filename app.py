import numpy as np
import csv
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

app = Flask(__name__)

# Load the dataset
with open(r'symptoms_data (3).csv') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    data = [row for row in reader]

# Define selected symptoms
selected_symptoms = [
    'Shortness', 'Shortness of breath', 'Runny or stuffy nose',
    'Increased sensitivity to cold', 'Weakness', 'Dizziness or lightheadedness',
    'Mild fever', 'Fever', 'Dizziness', 'Unintentional weight loss',
    'Headache', 'Nausea or vomiting', 'Nausea'
]

# Extracting features (symptoms) and labels (disease and drug)
X = np.array([row[:-2] for row in data], dtype=int)
y_disease = np.array([row[-2] for row in data])
y_drug = np.array([row[-1] for row in data])

# Encode labels
label_encoder_disease = LabelEncoder()
label_encoder_drug = LabelEncoder()
y_disease_encoded = label_encoder_disease.fit_transform(y_disease)
y_drug_encoded = label_encoder_drug.fit_transform(y_drug)

# Ensure labels are consistent in both training and test sets
def check_labels_consistency(y_train, y_test):
    train_labels = set(y_train)
    test_labels = set(y_test)
    if not test_labels.issubset(train_labels):
        raise ValueError(f"Test set contains unseen labels: {test_labels - train_labels}")

# Split data into train and test sets with stratification
X_train, X_test, y_disease_train, y_disease_test, y_drug_train, y_drug_test = train_test_split(
    X, y_disease_encoded, y_drug_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded)

# Check for label consistency
check_labels_consistency(y_disease_train, y_disease_test)
check_labels_consistency(y_drug_train, y_drug_test)

# Define the neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder_disease.classes_), activation='softmax')  # Output layer for disease prediction
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_disease_train, epochs=10, batch_size=32, validation_split=0.1)

# Function to predict disease and recommend one drug
def predict_disease_and_recommend_one_drug(symptoms):
    # Convert user input to binary values
    user_input = np.array([1 if symptom.lower() == 'yes' else 0 for symptom in symptoms])
    # Predict disease probabilities
    disease_probabilities = model.predict(user_input.reshape(1, -1))
    # Find the index of the disease with the highest probability
    predicted_disease_index = np.argmax(disease_probabilities)
    predicted_disease = label_encoder_disease.inverse_transform([predicted_disease_index])[0]
    # Recommend one drug
    drug_indices = np.where(y_disease_encoded == predicted_disease_index)[0]
    if len(drug_indices) == 0:
        return predicted_disease, "No drug found"
    recommended_drug_index = drug_indices[0]
    recommended_drug = label_encoder_drug.inverse_transform([recommended_drug_index])[0]
    return predicted_disease, recommended_drug

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json['symptoms']
    if all(symptom == 'no' for symptom in symptoms):
        return jsonify({'disease': 'unknown', 'drug': 'none'})
    else:
        try:
            predicted_disease, recommended_drug = predict_disease_and_recommend_one_drug(symptoms)
            return jsonify({'disease': predicted_disease, 'drug': recommended_drug})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
