# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:20:21 2018

@author: 20175876
"""
import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.title = 'Diabetic Retinopathy prediction'

app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

app.layout = html.Div(
    html.Div([
        html.Div([
                html.Div([
                        html.H1(
                                children='Diabetic Retinopathy prediction'),

                        html.Div(
                                children='''
                                Hello doctors. Welcome to our interactive tool to visualize predictions by using our classification algorithm. Please upload one image of one of the patient eyes by clicking the "Upload Image" at the top right. The file name, upload date and the image will be showed to you automatically.

                                '''),
                        html.Div(
                                children='''
                                Below you can find two bar graphs. The first bar graph shows a binary classification. This means: the probability that the patient is not sick or sick, based on our classification algorithm.
                                The second bar graph shows a multiclass classification. This bar graph shows in which stage the patient is most probably located, again based on our classification algorithm.
                                '''),
                        html.Div([
                                dcc.Upload(
                                        id="plot_buttton", children=html.Button("Choose model"),
                                        style={
                                            'width': '100%',
                                            'height': '80px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'none',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'marginTop': '0px'
                                            })
                                ]),

                        ],
                    style={'marginRight':'40px', 'marginLeft': '40px'},
                    className='nine columns'),

                html.Div([
                    dcc.Upload(
                        id='upload-image',
                        children=[html.Button('Upload Image'),
                        html.Hr()],
                        style={
                                'width': '100%',
                                'height': '80px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'none',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'marginRight': '10px'
                            },
                            multiple=True
                            ),
                    html.Div(id='output-image-upload'),
                ], className='two columns'),
            ],
        className="row"
        ),

        html.Div(
            [
            html.Div([
                dcc.Graph(
                    id='binary-graph',
                    figure={
                        'data': [
                            {'x': ["Not sick", "Sick"], 'y': [0.25,0.76], 'type': 'bar', 'marker':{'color': [0,1] , 'colorscale': 'RdBu', 'reversescale': False}}
                        ],
                        'layout': {
                            'title': 'Binary Classification',
                            'xaxis' : dict(
                                title='Classification',
                                titlefont=dict(
                                family='Helvetica, monospace',
                                size=20,
                                color='#7f7f7f'
                            )),
                            'yaxis' : dict(
                                title='Probability',
                                titlefont=dict(
                                family='Helvetica, monospace',
                                size=20,
                                color='#7f7f7f'
                            ))
                        }
                    }
                )
                ], className= 'six columns'
                ),

                html.Div([
                dcc.Graph(
                    id='example-graph-2',
                    figure={
                        'data': [
                            {'x': ['0: No', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative'], 'y': [0.1,0.3,0.1,0.35,0.15], 'type': 'bar', 'marker':{'color': [0,1,2,3,4] , 'colorscale': 'Reds', 'reversescale': False}}
                                ],
                        'layout': {
                            'title': 'Multiclass classification',
                            'xaxis' : dict(
                                title='Stages',
                                titlefont=dict(
                                family='Helvetica, monospace',
                                size=20,
                                color='#7f7f7f'
                            )),
                            'yaxis' : dict(
                                title='Probability',
                                titlefont=dict(
                                family='Helvetica, monospace',
                                size=20,
                                color='#7f7f7f'
                            ))
                        }
                    }
                )
                ], className= 'six columns'
                )
            ], className="row"
        )
    ], className='ten columns offset-by-one')
)

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, height=200, width=200),
        html.Hr(),
    ])



@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
