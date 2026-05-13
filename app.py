from flask import Flask, render_template, request, redirect, url_for

import pandas as pd
import pickle
from reportlab.pdfgen import canvas

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("cleaned_data.csv")

# =========================
# GET LOCATIONS
# =========================

locations = sorted(df['location'].unique())

# =========================
# LOAD TRAINED MODEL
# =========================

model = pickle.load(
    open("house_price_model.pkl", "rb")
)

# =========================
# LOGIN PAGE
# =========================

@app.route('/login', methods=['GET', 'POST'])

def login():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        if username == "admin" and password == "admin123":

            return redirect(url_for('home'))

    return render_template('login.html')

# =========================
# HOME PAGE
# =========================

@app.route('/home', methods=['GET', 'POST'])

def home():

    prediction = None

    location = ""
    sqft = ""
    bath = ""
    bhk = ""

    if request.method == 'POST':

        # GET FORM DATA

        sqft = float(request.form['total_sqft'])

        bath = float(request.form['bath'])

        bhk = float(request.form['bhk'])

        location = request.form['location']

        # CREATE INPUT DATA

        input_data = {
            'total_sqft': sqft,
            'bath': bath,
            'BHK': bhk
        }

        # ADD LOCATION COLUMNS

        for loc in locations:

            input_data[loc] = 0

        # SELECTED LOCATION = 1

        if location in input_data:

            input_data[location] = 1

        # CREATE DATAFRAME

        sample = pd.DataFrame([input_data])

        # MODEL PREDICTION

        result = model.predict(sample)

        prediction = float(result[0])

        # SAVE PREDICTION HISTORY

        with open("history.txt", "a") as file:

            file.write(

                f"Location: {location}, "

                f"Sqft: {sqft}, "

                f"Bath: {bath}, "

                f"BHK: {bhk}, "

                f"Predicted Price: {prediction}Lakhs\n"

            )

    return render_template(

        'index.html',

        prediction=prediction,

        locations=locations,

        location=location,

        sqft=sqft,

        bath=bath,

        bhk=bhk

    )

# =========================
# RUN APP
# =========================

@app.route('/download')
def download_pdf():

    c = canvas.Canvas("prediction_report.pdf")

    c.setFont("Helvetica-Bold", 18)

    c.drawString(120, 800, "House Price Prediction Report")

    c.save()

    return "PDF Report Generated Successfully!"


if __name__ == '__main__':
    app.run(debug=True)

