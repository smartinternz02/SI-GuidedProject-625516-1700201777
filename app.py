from flask import Flask, render_template, request
import joblib
from model import predict_function

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    ap = request.form['Application Type']
    ss =request.form['Signal Strength'] 
    l =request.form['Latency']
    rb =request.form['Required Bandwidth']
    ab=request.form['Allocated Bandwidth']
    input_data=[[ap,ss,l,rb,ab]]

    # Make prediction using your model
    prediction = predict_function(model, input_data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
