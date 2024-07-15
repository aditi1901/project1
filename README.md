# GREENTHREADS

#Sustainable Clothing Site: Our Solution 

Our solution centers on the creation of a website ‘Green Threads’ where users can access environmental performance scores for fashion brands, empowering them to make informed, sustainable purchasing decisions. Through this platform, users will earn reward points based on the sustainability of their purchases, incentivizing eco-friendly consumption habits. By providing transparent data on brands' environmental practices and rewarding sustainable choices, our website aims to drive awareness and adoption of sustainable fashion practices, contributing to a reduction in the fashion industry's environmental footprint and fostering a more conscientious consumer culture globally


# Sustainable Clothing Site: Future ML Model Integration

## Overview
##(This is something we aim to accomplish in the near future but weren't able to integrate right now due to time-constraints)

This Flask application demonstrates how a machine learning model can be integrated into our sustainable clothing site in the future to predict sustainability points for clothing items based on their features. The application includes functionality to train the model, save it, load it, and use it for predictions. This solution is intended to replace the current JavaScript if-else statements used for generating sustainability points.

## Features

- *Model Training:* Trains a Random Forest Regressor model using provided clothing data.
- *Prediction API:* Provides an endpoint to predict sustainability points for a given clothing item.
- *Data Preprocessing:* Handles categorical and numerical features, and boolean conversions.

## How It Works

### Model Training

The script includes a function train_model that:
1. Loads the dataset (sustainable_clothing_data.csv).
2. Preprocesses the data, including feature scaling and one-hot encoding.
3. Trains a Random Forest Regressor model.
4. Evaluates the model and prints Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
5. Saves the trained model as sustainable_clothing_model.pkl.

### Flask Application

The Flask application includes the following endpoints:
- /: Serves the main webpage (index.html).
- /get_clothing_data: Returns the clothing dataset as JSON.
- /predict: Accepts a POST request with a clothing item ID and returns the predicted sustainability points.

### How to Run

1. *Install Dependencies:*
   bash
   pip install flask pandas scikit-learn joblib
   

2. *Prepare the Data:*
   Ensure sustainable_clothing_data.csv is in the same directory as the script.

3. *Run the Application:*
   bash
   python app.py
   

4. *Access the Application:*
   Open your web browser and navigate to http://127.0.0.1:5000.

## Future Integration

*Note:* This ML model integration is planned for the future and is not currently implemented on the live site.

To integrate this ML model into our existing site in the future:
1. *Replace JavaScript Logic:*
   Replace the current JavaScript if-else statements for generating sustainability points with API calls to the Flask application.
   
2. *API Usage:*
   - Send a POST request to /predict with the clothing item ID.
   - Receive the predicted sustainability points and display them on your site.

### Example Integration Code (For Future Use)

#### JavaScript Example
javascript
fetch('/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ cloth_id: 1 })
})
.then(response => response.json())
.then(data => {
    console.log('Predicted Sustainability Points:', data['Predicted Sustainability Points']);
    // Update the site with the received points
});


#### HTML Example
html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sustainable Clothing</title>
</head>
<body>
    <div id="item-1">
        <h2>Item Name: Eco-Friendly Shirt</h2>
        <p id="sustainability-points"></p>
    </div>
    <script>
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ cloth_id: 1 })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('sustainability-points').textContent = 'Sustainability Points: ' + data['Predicted Sustainability Points'];
        });
    </script>
</body>
</html>


This integration will automate the sustainability points calculation and ensure more accurate and reliable results in the future.

---

This README clearly states that the ML model integration is planned for the future and provides an overview of the proposed changes and how they will enhance the site's functionality.




from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load

app = Flask(_name_)

# Function to train the model and save it
def train_model():
    # Load dataset
    df = pd.read_csv('sustainable_clothing_data.csv')

    # Separate features and target
    X = df.drop(columns=['Cloth Code', 'Sustainability Points'])
    y = df['Sustainability Points']

    # List of categorical and numerical features
    categorical_features = ['Item Name', 'Type', 'Material', 'Brand', 'Recyclability', 'Biodegradability', 'End-of-Life Options']
    numerical_features = ['Carbon Emissions (kg CO2e)', 'Water Usage (liters)', 'Energy Consumption (kWh)']

    # Convert 'Certifications' to a count of certifications
    X['Certifications'] = X['Certifications'].apply(lambda x: len(eval(x)))

    # Convert boolean columns to integers
    X['Recyclability'] = X['Recyclability'].astype(int)
    X['Biodegradability'] = X['Biodegradability'].astype(int)
    X['End-of-Life Options'] = X['End-of-Life Options'].astype(int)

    # Add 'Certifications' to the list of numerical features
    numerical_features.append('Certifications')

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Save the trained model
    dump(model_pipeline, 'sustainable_clothing_model.pkl')

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")

# Train the model when the script is run
train_model()

# Load the trained model
model = load('sustainable_clothing_model.pkl')

# Load the dataset
df = pd.read_csv('sustainable_clothing_data.csv')
clothing_data = df.to_dict(orient='index')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_clothing_data', methods=['GET'])
def get_clothing_data():
    return jsonify(clothing_data)

@app.route('/predict', methods=['POST'])
def predict():
    cloth_id = int(request.json['cloth_id'])
    data = clothing_data[cloth_id]

    # Preprocess the data
    data['Certifications'] = len(eval(data['Certifications']))
    data['Recyclability'] = int(data['Recyclability'])
    data['Biodegradability'] = int(data['Biodegradability'])
    data['End-of-Life Options'] = int(data['End-of-Life Options'])

    df = pd.DataFrame([data])
    df = df[['Item Name', 'Type', 'Material', 'Brand', 'Carbon Emissions (kg CO2e)',
             'Water Usage (liters)', 'Energy Consumption (kWh)', 'Certifications',
             'Recyclability', 'Biodegradability', 'End-of-Life Options']]

    # Predict the sustainability points
    prediction = model.predict(df)

    return jsonify({'Predicted Sustainability Points': prediction[0]})

if _name_ == '_main_':
    app.run(debug=True)


