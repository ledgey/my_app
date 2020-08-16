import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import helper_functions as hf
import pandas as pd

import flask



server = flask.Flask(__name__) # define flask app.server


app = dash.Dash(__name__, server=server)


ticks = ['GBPUSD=X', 'GBPEUR=X', 'EURUSD=X', 'GBPJPY=X', 'JPY=X', 'GBP=X', 'GBPAUD=X', 'GBPBRL=X', 'GBPCAD=X', 'GBPCHF=X']

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
    html.Div([
            dcc.Dropdown(
                id='ticker',
                options=[{'label': i, 'value': i} for i in ticks],
                value='EURUSD=X'
            ),
        ],
        style={'width': '30%', 'display': 'inline-block'}),
            html.Div(id='timestamp'),
                html.Div([
                    html.Div([
                        dcc.Graph(id='indicator-graphic')],
                        className="eight columns"),
                html.Div([
                        dcc.Graph(id='dist-graph')],
                        className="four columns")],
                className="row")
])

@app.callback(
    [Output('indicator-graphic', 'figure'),
     Output('dist-graph', 'figure'),
     Output(component_id='timestamp', component_property='children')],
    [Input('ticker', 'value')])
def update_graph(ticker):
    fig, dist, time = hf.runPrediction(ticker)

    return fig, dist, 'Timestamp: {}'.format(time)


if __name__ == '__main__':
    app.run_server()