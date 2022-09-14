import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,static_url_path='/static')
model = pickle.load(open('SP_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Absence = request.form['absences']
    G1 = request.form['G1']
    G2 = request.form['G2']
    prediction = model.predict([[Absence, G1, G2]])
    print("final features",prediction)
    print("prediction:",prediction)
    output = round(prediction[0])
    print(output)
    return render_template('index.html',prediction_text='Your Grade 3 Score  {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)