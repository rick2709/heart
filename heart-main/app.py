from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import tensorflow.keras

app = Flask(__name__)

model = tensorflow.keras.models.load_model('heartmodel.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    age =request.form['age'] 
    resting_blood_pressure =request.form["resting_blood_pressure"]   
    cholesterol =request.form["cholesterol"]
    fasting_blood_sugar =request.form["fasting_blood_sugar"]
    
    max_heart_rate_achieved =request.form["max_heart_rate_achieved"]
    
    exercise_induced_angina =request.form["exercise_induced_angina"]
    
    sex =request.form["sex"]
    values = [age, resting_blood_pressure, cholesterol, fasting_blood_sugar,
              max_heart_rate_achieved, exercise_induced_angina, sex]
    
		
   
    scaler = MinMaxScaler()
    values = scaler.fit_transform(np.reshape(values, (1, -1)))
    output = model.predict_classes(values)
    
    if output == 1:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "
        

    return render_template('result.html', prediction_text='Patient has {}'.format(res_val))
    
if __name__ == "__main__":
    app.run(debug=True)
