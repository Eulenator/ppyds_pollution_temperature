from flask import Flask
from flask import render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# flask get values from user for ml model prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    models = pd.read_pickle("model.pkl")
    countries = models['Entity'].to_list()
    if request.method == 'POST':
        # get values from user
        values = request.form
        values_list = list(values.values())
        values_array = np.array(values_list)
        input_x = values_array[:-1]
        model_country = models[models['Entity'] == values_list[-1]]
        try:
            prediction = model_country.iloc[0]["Temperature Change Model"].predict([input_x])
        except:
            return render_template('index.html', result='Error: Invalid input! Cannot predict.', countries=countries)
        return render_template('index.html', result='The predicted value is {}'.format(prediction[0]), countries=countries)
    return render_template('index.html', countries=countries)

# create flask main function
if __name__ == '__main__':
    app.run(debug=True)