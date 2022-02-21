import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    item = [x for x in request.form.values()]
    data = []

    data.append(int(item[0]))
    if item[1] == 'Male':
        data.append(0)
        data.append(1)
    else:
        data.append(1)
        data.append(0)

    if item[2] == 'No':
        data.append(1)
        data.append(0)
    else:
        data.append(0)
        data.append(1)
    
    prediction = model.predict([data])

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The Insurance cost will be   $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)