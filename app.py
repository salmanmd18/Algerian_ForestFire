import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('AlgerianClassifier.pickle', 'rb'))

@app.route('/predict_fire',methods= ['POST'])
def predict_fire():

    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = int(model.predict(new_data)[0])
    if output == 1:
        text = "Forest is in danger"
    else:
        text = "Forest is safe"
    return jsonify(text,output)

if __name__ == "__main__":
    app.run(debug=True)