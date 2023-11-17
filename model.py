import numpy as np

def predict_function(model, input_data):
    input_data_2d = np.array(input_data).reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_2d)

    return prediction
