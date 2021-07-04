import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

with open('R178500B.ipynb','rb') as pickle_file:
    model=pickle.load(pickle_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "resting_blood_pressure","cholesterol","fasting_blood_sugar", "max_heart_rate_achieved" ," exercise_induced_angina","st_depression", "sex", "chest_pain_type_angina","chest_pain_type_non_angina","chest_pain_type_typical_agina", "rest_ecg","rest_ecg_normal","st_slope","enter st_slope__normal"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
    
if __name__ == "__main__":
    app.run()
