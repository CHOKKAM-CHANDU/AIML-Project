from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb  # Assuming you're using XGBoost for your machine learning model

app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model('FoodDelivery.pkl')

# Define a function to preprocess input data before passing it to the model
def preprocess_input(input_data):
    # Perform any necessary preprocessing steps here, such as scaling or encoding
    # Return the preprocessed input data as a numpy array
    return np.array(input_data)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle the form submission and make predictions
# Define a route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    id = float(request.form['id'])
    delivery_person_age = float(request.form['delivery_person_age'])
    delivery_person_ratings = float(request.form['delivery_person_ratings'])
    restaurant_latitude = float(request.form['restaurant_latitude'])
    restaurant_longitude = float(request.form['restaurant_longitude'])
    delivery_location_latitude = float(request.form['delivery_location_latitude'])
    delivery_location_longitude = float(request.form['delivery_location_longitude'])
    vehicle_condition = float(request.form['vehicle_condition'])
    multiple_deliveries = float(request.form['multiple_deliveries'])
    weather_conditions_numerical = float(request.form['weather_conditions_numerical'])
    type_of_order_numerical = float(request.form['type_of_order_numerical'])
    type_of_vehicle_numerical = float(request.form['type_of_vehicle_numerical'])
    festival_numerical = float(request.form['festival_numerical'])
    city_numerical = float(request.form['city_numerical'])
    road_traffic_density_numerical = float(request.form['road_traffic_density_numerical'])
    day = float(request.form['day'])
    month = float(request.form['month'])
    quarter = float(request.form['quarter'])
    year = float(request.form['year'])
    day_of_week = float(request.form['day_of_week'])
    is_month_start = float(request.form['is_month_start'])
    is_month_end = float(request.form['is_month_end'])
    is_quarter_start = float(request.form['is_quarter_start'])
    is_quarter_end = float(request.form['is_quarter_end'])
    is_year_start = float(request.form['is_year_start'])
    is_year_end = float(request.form['is_year_end'])
    is_weekend = float(request.form['is_weekend'])
    order_prepare_time = float(request.form['order_prepare_time'])
    distance = float(request.form['distance'])

    # Preprocess the input data
    input_data = preprocess_input([id, delivery_person_age, delivery_person_ratings, restaurant_latitude, restaurant_longitude, 
                                   delivery_location_latitude, delivery_location_longitude, vehicle_condition, multiple_deliveries,
                                   weather_conditions_numerical, type_of_order_numerical, type_of_vehicle_numerical,
                                   festival_numerical, city_numerical, road_traffic_density_numerical, day, month, quarter,
                                   year, day_of_week, is_month_start, is_month_end, is_quarter_start, is_quarter_end,
                                   is_year_start, is_year_end, is_weekend, order_prepare_time, distance])

    # Make a prediction using the preprocessed input data
    prediction = model.predict(input_data.reshape(1, -1))

    # Render the prediction result template with the prediction value
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
