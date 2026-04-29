import gradio as gr
import pandas as pd
import joblib

# Loading Model
model = joblib.load("flight_fare_model_fast.joblib")

def predict_fare(airline, source, dest, dep_time, arr_time, travel_class, duration, days_left, stops):
    stops_map = {"Zero": 0, "One": 1, "Two or More": 2}
    
    is_urgent = 1 if days_left <= 2 else 0
    
    input_df = pd.DataFrame([[
        airline, source, dest, dep_time, arr_time, travel_class, 
        duration, days_left, stops_map[stops], is_urgent
    ]], columns=['airline', 'source_city', 'destination_city', 'departure_time', 
                 'arrival_time', 'class', 'duration', 'days_left', 'stops', 'is_urgent'])
    
    prediction = model.predict(input_df)[0]
    return f"Estimated Fare: {round(prediction, 2)}"


iface = gr.Interface(
    fn=predict_fare,
    inputs=[
        gr.Dropdown(["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"], label="Airline"),
        gr.Dropdown(["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], label="Source City"),
        gr.Dropdown(["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"], label="Destination City"),
        gr.Dropdown(["Evening", "Early_Morning", "Morning", "Afternoon", "Night", "Late_Night"], label="Departure Time"),
        gr.Dropdown(["Evening", "Early_Morning", "Morning", "Afternoon", "Night", "Late_Night"], label="Arrival Time"),
        gr.Radio(["Economy", "Business"], label="Class"),
        gr.Number(label="Duration (Hours)"),
        gr.Number(label="Days Left Until Flight"),
        gr.Radio(["Zero", "One", "Two or More"], label="Stops")
    ],
    outputs="text",
    title="Flight Fare Predictor"
)

iface.launch()