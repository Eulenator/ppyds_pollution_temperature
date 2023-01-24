from flask import Flask
from flask import render_template, request
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import json


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    models = pd.read_pickle("df_models_per_country.pkl")
    df_historic = pd.read_pickle("df_emmissions_temperatures.pkl")
    df_historic_temperatures = df_historic[['Year', 'Entity', 'Temperature Change']]
    countries = models['Entity'].to_list()

    if request.method == 'POST':
        values = request.form
        values_list = list(values.values())
        values_array = np.array(values_list)
        input_x = values_array[:-1]
        model_country = models[models['Entity'] == values_list[-1]]
        try:
            prediction = _predict(input_x, model_country)
        except:
            return render_template('index.html', result='Error: Invalid input! Cannot predict.', countries=countries)
        return render_template('index.html',
         result='The predicted value is {}'.format(prediction),
          countries=countries, 
          graphData=_create_plot(_get_last_temperatures_based_on_pred(df_historic_temperatures, values_list[-1], int(values_list[-2]), prediction)))
    return render_template('index.html', countries=countries)


def _predict(input_x, model_country):
    print(input_x[-2])
    prediction = model_country.iloc[0]["Temperature Change Model"].predict([input_x])
    return prediction[0]


def _get_last_temperatures_based_on_pred(df_historic_temperatures, country, prediction_year, prediction):
    df_historic_temperatures = df_historic_temperatures[df_historic_temperatures['Entity'] == country]
    df_historic_temperatures = df_historic_temperatures[df_historic_temperatures['Year'] < prediction_year]
    df_historic_temperatures = df_historic_temperatures.sort_values(by=['Year'], ascending=False)
    df_historic_temperatures = df_historic_temperatures.head(5)
    df_historic_temperatures = df_historic_temperatures.append({'Year': prediction_year
        , 'Entity': country
        , 'Temperature Change': prediction}, ignore_index=True)
    return df_historic_temperatures.sort_values(['Entity', 'Year'])

def _create_plot(df_historic_temperatures):
    data = [
        go.Line(
            x=df_historic_temperatures['Year'],
            y=df_historic_temperatures['Temperature Change']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON



if __name__ == '__main__':
    app.run(debug=True)