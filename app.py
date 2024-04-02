from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)


# Load your model
model = joblib.load('finalized_random_forest_model.pkl')

@app.route('/')
def index():
    # Serve the HTML form
    return render_template('design.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    q1 = float(data['features']['q1Msec'])
    q2 = float(data['features']['q2Msec'])
    q3 = float(data['features']['q3Msec'])
    
    # Calculate meanPace and maxPace, considering only non-zero values
    q_times = [q1, q2, q3]
    non_zero_q_times = [time for time in q_times if time != 0]
    meanPace = np.mean(non_zero_q_times) if non_zero_q_times else 0
    maxPace = np.max(non_zero_q_times) if non_zero_q_times else 0
    
    data['features']['meanPace'] = meanPace
    data['features']['maxPace'] = maxPace
    
    feature_names = ['driverId', 'qualiResultPosition', 'meanPace', 'maxPace', 'driverExpYears', 'grid_penalty']
    
    # Create a DataFrame for prediction
    # Assuming the order of values matches feature_names
    features_data = [[data['features'][feature] for feature in feature_names]]
    features_df = pd.DataFrame(features_data, columns=feature_names)
    
    # Prediction logic using the model...
    # Be sure 'data['features']' matches the shape and data type expected by your model
    prediction = model.predict(features_df)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
