#Importing the Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, neighbors,tree
import pandas as pd

# Reading the Data
data = pd.read_csv('./Dataset/battery_data.csv')

data1 = pd.read_csv('./Dataset/B0029-Discharge.csv')

#Splitting Input  and Target Features
X = data[['Charge_Max_Time','Discharge_Max_Time','Charging_Max_Threshold_Voltage_time','Discharging_Min_Threshold_Voltage_time']].values
y=data['Capacity'].values

X1=data[['Charge_Max_Time','Discharge_Max_Time','Charging_Max_Threshold_Voltage_time','Discharging_Min_Threshold_Voltage_time','Capacity']]
y1=data['Cycle_Prediction'].values

#Performing Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.25, random_state=34)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1,test_size=0.25, random_state=34)


#Creating three different models
models = {'Linear Regression': linear_model.LinearRegression,
          'Decision Tree Regression': tree.DecisionTreeRegressor,
          'K Nearest Neighbors': neighbors.KNeighborsRegressor}

models1 = {'Linear Regression': linear_model.LinearRegression,
          'Decision Tree Regression': tree.DecisionTreeRegressor,
          'K Nearest Neighbors': neighbors.KNeighborsRegressor}

# Battery Data figure
def voltage_curve():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.line(data1, x="Time", y="Voltage_measured", color="Cycle",
                                   title='Discharging voltage curve for different cycles',template='plotly_dark',
                                   labels=dict(Time='Time (Secs)',Voltage_measured='Voltage Measured (V)',Cycle='Cycles')
                                   ).add_annotation( 
                                       text="Charge-Cut-Off-Voltage", x=0, y=4.2, arrowhead=1, showarrow=True
                                       ).add_annotation( 
                                           text="Self-Recharge", x=1600, y=3.3, arrowhead=1, showarrow=True
                                           ).add_hline(y=2.0, line_dash="dot",
                                                       annotation_text="Discharge-Cut-Off-Voltage", 
                                                       annotation_position="bottom left").update_layout(
                                                           title_x=0.5,
                                                           plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           ),
                   
                ) 
            ])
        ),  
    ])

def temperature_curve():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.line(data1,x='Time',y='Temperature_measured',color='Cycle',
                                   title='Discharging temperature curve for different cycles',template='plotly_dark',
                                   labels=dict(Time='Time (Secs)',Temperature_measured='Temperature Measured (C)',Cycle='Cycles')
                                   ).add_annotation( 
                                       text="Discharge", x=700, y=53, arrowhead=1, showarrow=True
                                       ).add_annotation( 
                                           text="Self-Recharge", x=1490, y=60, arrowhead=1, showarrow=True
                                           ).update_layout(
                                                           title_x=0.5,
                                                           plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           ),
                    )
                ])
            ),
        ])
                                             
# Text field
def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Battery Visualization Dashboard"),
                ], style={'textAlign': 'center'}) 
            ])
        ),
    ])



# Build App
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText()
                ], width=12)
            ]), 
            html.Br(),
            dbc.Row([
                dbc.Col([
                    voltage_curve() 
                ], width=6),
                dbc.Col([
                   temperature_curve()
                ], width=6),
            ], align='center'), 
            html.Br(),
           dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='model-name',      
                        options=[{'label': x, 'value': x} 
                                 for x in models],
                        value='Linear Regression',
                       clearable=False,
                       style={ 'color': 'darkblue'}
                    ),
                    
                   dcc.Graph(id="graph"),
                ], width=6),
                
                dbc.Col([
                    dcc.Dropdown(
                        id='model-name1',      
                        options=[{'label': i, 'value': i} 
                                 for i in models1],
                        value='Linear Regression',
                       clearable=False,
                       style={ 'color': 'darkblue'}
                       #style={'backgroundColor': '#272B30', 'color': 'darkgreen'}
                    ),
                    
                   dcc.Graph(id="graph1"),
                ], width=6)
                
            ]),       
        ]), color = 'dark'
    )
])

@app.callback(
    Output("graph", "figure"),
    [Input('model-name', "value")])

def capacity_prediction(name):
    model=models[name]()
    model.fit(X_train,y_train)
    
    fig=px.line(x=data.Cycle,y=list(model.predict(X)),
                  labels=dict(x='Cycles',y='Predicted Remaining Capacities'),
                  title='Predicted Remaining Useful Capacities',template='plotly_dark',).update_layout(
                                                           title_x=0.5,
                                                           plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           )

    return fig

@app.callback(
    Output("graph1", "figure"),
    [Input('model-name1', "value")])

def cycle_prediction(name1):
    model1=models1[name1]()
    model1.fit(X_train1,y_train1)
    
    fig1=px.line(x=data.Cycle,y=list(model1.predict(X1)),
                  labels=dict(x='Cycles',y='Predicted Remaining Cycles'),
                  title='Predicted Remaining Useful Cycles',template='plotly_dark',).update_layout(
                                                           title_x=0.5,
                                                           plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                                           )

    return fig1


if __name__ == "__main__":
    app.run_server(debug=True)
