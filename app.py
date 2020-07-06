# Import required libraries
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
import lightgbm as lgb


# Multi-dropdown options
from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Create controls
county_options = [
    {"label": str(COUNTIES[county]), "value": str(county)} for county in COUNTIES
]

well_status_options = [
    {"label": str(WELL_STATUSES[well_status]), "value": str(well_status)}
    for well_status in WELL_STATUSES
]

well_type_options = [
    {"label": str(WELL_TYPES[well_type]), "value": str(well_type)}
    for well_type in WELL_TYPES
]


# Load data
df = pd.read_csv(DATA_PATH.joinpath("wellspublic.csv"), low_memory=False)
df["Date_Well_Completed"] = pd.to_datetime(df["Date_Well_Completed"])
df = df[df["Date_Well_Completed"] > dt.datetime(1960, 1, 1)]

trim = df[["API_WellNo", "Well_Type", "Well_Name"]]
trim.index = trim["API_WellNo"]
dataset = trim.to_dict(orient="index")

points = pickle.load(open(DATA_PATH.joinpath("points.pkl"), "rb"))
####################################""
test_corrs_removed = pd.read_csv(DATA_PATH.joinpath("test_bureau_corrs_removed.csv"), low_memory=False)

features = pd.read_csv(DATA_PATH.joinpath("features.csv"), low_memory=False)
features.drop('Unnamed: 0', axis=1, inplace=True)
test_features = pd.read_csv(DATA_PATH.joinpath("test_features.csv"), low_memory=False)
test_features.drop('Unnamed: 0', axis=1, inplace=True)

# test_corrs_removed = test_corrs_removed.sample(n=10, random_state=1)
# test_corrs_removed = test_corrs_removed.sort_values('SK_ID_CURR')

client_id = test_corrs_removed["SK_ID_CURR"].tolist()

#Load models

clf_non_solvable = load(DATA_PATH.joinpath('lgb_pos_weights.joblib'))
#clf_solvable = load(DATA_PATH.joinpath('lgb_neg_weights.joblib'))
clf_solvable = load('/home/marwan/Downloads/drive.joblib')

# #Preprocess data

# def preprocess_data(df):
#     test_features = df
#     test_features.fillna(test_features.median(), inplace=True)
#     test_ids = test_features['SK_ID_CURR']
#     test_features = test_features.drop(columns = ['SK_ID_CURR'])
#     test_features = pd.get_dummies(test_features)
#     return test_features

# test_features = preprocess_data(test_corrs_removed)

# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("logo.png"),
                            id="plotly-image",
                            style={
                                "height": "200",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Prêt à dépenser",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Société de crédit", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        # html.A(
                        #     html.Button("Learn More", id="learn-more-button"),
                        #     href="https://plot.ly/dash/pricing/",
                        # )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [   

                        html.P("Optimiser la recherche des clients :", className="control_label"),
                        
                        dcc.RadioItems(
                            id="well_status_selector",
                            options=[
                                {"label": "Solvable ", "value": "solvable"},
                                {"label": "Non solvable ", "value": "nonsolvable"},
                            ],
                            value="solvable",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        html.P(
                            "Choix du client :",
                            className="control_label",
                        ),
                        dcc.Dropdown(
                            id="client_id",
                            options=[{'label': i, 'value': i} for i in client_id],
                            multi=False,
                            value='116949',
                            placeholder='CLIENT_ID',                    
                            className="dcc_control",
                        ),
                        # html.Button("Chercher", id="submit"),           
                        # html.P(
                        #     "Choix du 1er critère",
                        #     className="control_label",
                        # ),             
                        dcc.Dropdown(
                            id="critere_1",
                            options=[{'label': i, 'value': i} for i in test_corrs_removed.columns.values.tolist()],
                            multi=False,
                            value='AMT_INCOME_TOTAL',
                            className="dcc_control",
                        ),
                        html.P(id='affichage_crit_1'),

                        html.P(
                            "Comparaison avec un autre client:",
                            className="control_label",
                        ),
                        dcc.Dropdown(
                            id="client_id2",
                            options=[{'label': i, 'value': i} for i in client_id],
                            multi=False,
                            value='116949',
                            placeholder='CLIENT_ID',                    
                            className="dcc_control",
                        ),
                        #html.P("Choix du 2nd critère :", className="control_label"),
                        dcc.Dropdown(
                            id="critere_2",
                            options=[{'label': i, 'value': i} for i in test_corrs_removed.columns.values.tolist()],
                            multi=False,
                            value='AMT_INCOME_TOTAL',
                            className="dcc_control",
                        ),
                        html.P(id='affichage_crit_2'),

                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="defaut_paimentText"), html.P("Probabilité de défaut de paiment")],
                                    id="defaut_paiment",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="ageText"), html.P("Age")],
                                    id="age",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="revenueText"), html.P("Revenues totals")],
                                    id="revenue",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="jobText"), html.P("Métier")],
                                    id="job",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [],#dcc.Graph(id="count_graph")
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [html.H6('Explication de la probabilité de defaut', style={'text-align':'center'}), 
                    dcc.Graph(id="main_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="individual_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        # html.Div(
        #     [
        #         html.Div(
        #             [dcc.Graph(id="pie_graph")],
        #             className="pretty_container seven columns",
        #         ),
        #         html.Div(
        #             [dcc.Graph(id="aggregate_graph")],
        #             className="pretty_container five columns",
        #         ),
        #     ],
        #     className="row flex-display",
        # ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)

@app.callback(
    Output('affichage_crit_1', 'children'),
    [Input('critere_1', 'value'),
     Input('client_id', 'value')]

)
def update_crit1(value, client_id):
    return 'Valeur : {}'.format(test_corrs_removed[value].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]])

@app.callback(
    Output('affichage_crit_2', 'children'),
    [Input('critere_2', 'value'),
     Input('client_id2', 'value')]

)
def update_crit1(value, client_id):
    return 'Valeur : {}'.format(test_corrs_removed[value].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]])



@app.callback(
    Output('ageText', 'children'),
    [Input('client_id', 'value')]
)

def update_age(client_id):
    age = test_corrs_removed['DAYS_BIRTH'].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]]
    return '{} ans'.format(abs(int(np.round(age/365, decimals=0))))


@app.callback(
    Output('revenueText', 'children'),
    [Input('client_id', 'value')]
)

def update_age(client_id):
    amt = test_corrs_removed['AMT_INCOME_TOTAL'].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]]
    return '{}'.format(int(np.round(amt, decimals=0)))

@app.callback(
    Output('jobText', 'children'),
    [Input('client_id', 'value')]
)

def update_age(client_id):
    travail = test_corrs_removed['OCCUPATION_TYPE'].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]]
  
    return '{}'.format(travail)



# Radio -> multi
@app.callback(
    Output('defaut_paimentText', 'children'), 
    #[Input("submit", "n_clicks")],
    [#Input("well_status_selector", "value"), 
    Input('client_id', 'value')]
)
def display_status(client_id):
    # if selector == "solvable":
    #clf = load(DATA_PATH.joinpath('lgb_pos_weights.joblib'))

    #clf = clf_solvable
    #     #return list(WELL_STATUSES.keys())
    # elif selector == "nonsolvable":
    #     clf = clf_non_solvable
        #return ["AC"]
    clf_solvable = load('data/lgb_neg_weights.joblib')

    #data_client = test_features.iloc[5].values.reshape(1, -1)
    defaut_precentage = clf_solvable.predict_proba(test_features.iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)]].values.reshape(1, -1))[0][1]

    print(defaut_precentage)
    return '{} %'.format(defaut_precentage*100)


@app.callback(
    Output('main_graph', 'figure'),
    [Input('client_id', 'value')]
)

def update_graph(client_id):

    test_features_filled= test_features.fillna(test_features.median())

    lime1 = LimeTabularExplainer(test_features_filled,
                                feature_names=test_features_filled.columns,
                                discretize_continuous=False)
                                

    exp = lime1.explain_instance(test_features_filled.iloc[1],
                                clf_solvable.predict_proba,
                                num_samples=1000)

    exp_list = exp.as_list()
    exp_keys = []
    exp_values = []
    exp_positives = []
    for i in range(len(exp_list)):
        exp_keys.append(exp_list[i][0])
        exp_values.append(exp_list[i][1])
    # if exp_values[i] <= 0:
    #   exp_positives.append('green')
    # elif exp_values[i] > 0:
    #   exp_positives.append('red')

    df_data = pd.DataFrame(data=[exp_keys,exp_values])
    df_data = df_data.T
    df_data.columns=['exp_keys', 'exp_values']
    df_data = df_data.iloc[np.abs(df_data['exp_values'].values).argsort()]
    df_data['color'] = df_data.exp_values.apply(lambda x: 'red' if x > 0 else 'green')
    fig = go.Figure(go.Bar(
                x=df_data['exp_values'],
                y=df_data['exp_keys'],
                orientation='h',
                marker_color=df_data['color'])            
                )

  
    return fig



# # Helper functions
# def human_format(num):
#     if num == 0:
#         return "0"

#     magnitude = int(math.log(num, 1000))
#     mantissa = str(int(num / (1000 ** magnitude)))
#     return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


# def filter_dataframe(df, well_statuses, well_types, year_slider):
#     dff = df[
#         df["Well_Status"].isin(well_statuses)
#         & df["Well_Type"].isin(well_types)
#         & (df["Date_Well_Completed"] > dt.datetime(year_slider[0], 1, 1))
#         #& (df["Date_Well_Completed"] < dt.datetime(year_slider[1], 1, 1))
#     ]
#     return dff


# def produce_individual(api_well_num):
#     try:
#         points[api_well_num]
#     except:
#         return None, None, None, None

#     index = list(
#         range(min(points[api_well_num].keys()), max(points[api_well_num].keys()) + 1)
#     )
#     gas = []
#     oil = []
#     water = []

#     for year in index:
#         try:
#             gas.append(points[api_well_num][year]["Gas Produced, MCF"])
#         except:
#             gas.append(0)
#         try:
#             oil.append(points[api_well_num][year]["Oil Produced, bbl"])
#         except:
#             oil.append(0)
#         try:
#             water.append(points[api_well_num][year]["Water Produced, bbl"])
#         except:
#             water.append(0)

#     return index, gas, oil, water


# #def produce_aggregate(selected, year_slider):

#     index = list(range(max(year_slider[0], 1985), 2016))
#     gas = []
#     oil = []
#     water = []

#     for year in index:
#         count_gas = 0
#         count_oil = 0
#         count_water = 0
#         for api_well_num in selected:
#             try:
#                 count_gas += points[api_well_num][year]["Gas Produced, MCF"]
#             except:
#                 pass
#             try:
#                 count_oil += points[api_well_num][year]["Oil Produced, bbl"]
#             except:
#                 pass
#             try:
#                 count_water += points[api_well_num][year]["Water Produced, bbl"]
#             except:
#                 pass
#         gas.append(count_gas)
#         oil.append(count_oil)
#         water.append(count_water)

#     return index, gas, oil, water


# # Create callbacks
# app.clientside_callback(
#     ClientsideFunction(namespace="clientside", function_name="resize"),
#     Output("output-clientside", "children"),
#     [Input("count_graph", "figure")],
# )


# @app.callback(
#     Output("aggregate_data", "data"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def update_production_text(well_statuses, well_types, year_slider):

#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)
#     selected = dff["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)
#     return [human_format(sum(gas)), human_format(sum(oil)), human_format(sum(water))]


# # Radio -> multi
# @app.callback(
#     Output("well_statuses", "value"), [Input("well_status_selector", "value")]
# )
# def display_status(selector):
#     if selector == "all":
#         return list(WELL_STATUSES.keys())
#     elif selector == "active":
#         return ["AC"]
#     return []


# # Radio -> multi
# @app.callback(Output("well_types", "value"), [Input("well_type_selector", "value")])
# def display_type(selector):
#     if selector == "all":
#         return list(WELL_TYPES.keys())
#     elif selector == "productive":
#         return ["GD", "GE", "GW", "IG", "IW", "OD", "OE", "OW"]
#     return []


# # Slider -> count graph
# @app.callback(Output("year_slider", "value"), [Input("count_graph", "selectedData")])
# def update_year_slider(count_graph_selected):

#     if count_graph_selected is None:
#         return [1990, 2010]

#     nums = [int(point["pointNumber"]) for point in count_graph_selected["points"]]
#     return [min(nums) + 1960, max(nums) + 1961]


# # Selectors -> well text
# @app.callback(
#     Output("well_text", "children"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def update_well_text(well_statuses, well_types, year_slider):

#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)
#     return dff.shape[0]


# @app.callback(
#     [
#         Output("gasText", "children"),
#         Output("oilText", "children"),
#         Output("waterText", "children"),
#     ],
#     [Input("aggregate_data", "data")],
# )
# def update_text(data):
#     return data[0] + " mcf", data[1] + " bbl", data[2] + " bbl"


# # Selectors -> main graph
# @app.callback(
#     # Output("main_graph", "figure"),
#     # [
#     #     Input("well_statuses", "value"),
#     #     Input("well_types", "value"),
#     #     Input("year_slider", "value"),
#     # ],
#     # [State("lock_selector", "value"), State("main_graph", "relayoutData")],
# )
# def make_main_figure(
#     well_statuses, well_types, year_slider, selector, main_graph_layout
# ):

#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)

#     traces = []
#     for well_type, dfff in dff.groupby("Well_Type"):
#         trace = dict(
#             type="scattermapbox",
#             lon=dfff["Surface_Longitude"],
#             lat=dfff["Surface_latitude"],
#             text=dfff["Well_Name"],
#             customdata=dfff["API_WellNo"],
#             name=WELL_TYPES[well_type],
#             marker=dict(size=4, opacity=0.6),
#         )
#         traces.append(trace)

#     # relayoutData is None by default, and {'autosize': True} without relayout action
#     if main_graph_layout is not None and selector is not None and "locked" in selector:
#         if "mapbox.center" in main_graph_layout.keys():
#             lon = float(main_graph_layout["mapbox.center"]["lon"])
#             lat = float(main_graph_layout["mapbox.center"]["lat"])
#             zoom = float(main_graph_layout["mapbox.zoom"])
#             layout["mapbox"]["center"]["lon"] = lon
#             layout["mapbox"]["center"]["lat"] = lat
#             layout["mapbox"]["zoom"] = zoom

#     figure = dict(data=traces, layout=layout)
#     return figure


# # Main graph -> individual graph
# @app.callback(Output("individual_graph", "figure"), [Input("main_graph", "hoverData")])
# def make_individual_figure(main_graph_hover):

#     layout_individual = copy.deepcopy(layout)

#     if main_graph_hover is None:
#         main_graph_hover = {
#             "points": [
#                 {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
#             ]
#         }

#     chosen = [point["customdata"] for point in main_graph_hover["points"]]
#     index, gas, oil, water = produce_individual(chosen[0])

#     if index is None:
#         annotation = dict(
#             text="No data available",
#             x=0.5,
#             y=0.5,
#             align="center",
#             showarrow=False,
#             xref="paper",
#             yref="paper",
#         )
#         layout_individual["annotations"] = [annotation]
#         data = []
#     else:
#         data = [
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Gas Produced (mcf)",
#                 x=index,
#                 y=gas,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#fac1b7"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Oil Produced (bbl)",
#                 x=index,
#                 y=oil,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#a9bb95"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#             dict(
#                 type="scatter",
#                 mode="lines+markers",
#                 name="Water Produced (bbl)",
#                 x=index,
#                 y=water,
#                 line=dict(shape="spline", smoothing=2, width=1, color="#92d8d8"),
#                 marker=dict(symbol="diamond-open"),
#             ),
#         ]
#         layout_individual["title"] = dataset[chosen[0]]["Well_Name"]

#     figure = dict(data=data, layout=layout_individual)
#     return figure


# # Selectors, main graph -> aggregate graph
# @app.callback(
#     Output("aggregate_graph", "figure"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#         Input("main_graph", "hoverData"),
#     ],
# )
# def make_aggregate_figure(well_statuses, well_types, year_slider, main_graph_hover):

#     layout_aggregate = copy.deepcopy(layout)

#     if main_graph_hover is None:
#         main_graph_hover = {
#             "points": [
#                 {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
#             ]
#         }

#     chosen = [point["customdata"] for point in main_graph_hover["points"]]
#     well_type = dataset[chosen[0]]["Well_Type"]
#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)

#     selected = dff[dff["Well_Type"] == well_type]["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)

#     data = [
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Gas Produced (mcf)",
#             x=index,
#             y=gas,
#             line=dict(shape="spline", smoothing="2", color="#F9ADA0"),
#         ),
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Oil Produced (bbl)",
#             x=index,
#             y=oil,
#             line=dict(shape="spline", smoothing="2", color="#849E68"),
#         ),
#         dict(
#             type="scatter",
#             mode="lines",
#             name="Water Produced (bbl)",
#             x=index,
#             y=water,
#             line=dict(shape="spline", smoothing="2", color="#59C3C3"),
#         ),
#     ]
#     layout_aggregate["title"] = "Aggregate: " + WELL_TYPES[well_type]

#     figure = dict(data=data, layout=layout_aggregate)
#     return figure


# # Selectors, main graph -> pie graph
# @app.callback(
#     Output("pie_graph", "figure"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def make_pie_figure(well_statuses, well_types, year_slider):

#     layout_pie = copy.deepcopy(layout)

#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)

#     selected = dff["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)

#     aggregate = dff.groupby(["Well_Type"]).count()

#     data = [
#         dict(
#             type="pie",
#             labels=["Gas", "Oil", "Water"],
#             values=[sum(gas), sum(oil), sum(water)],
#             name="Production Breakdown",
#             text=[
#                 "Total Gas Produced (mcf)",
#                 "Total Oil Produced (bbl)",
#                 "Total Water Produced (bbl)",
#             ],
#             hoverinfo="text+value+percent",
#             textinfo="label+percent+name",
#             hole=0.5,
#             marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
#             domain={"x": [0, 0.45], "y": [0.2, 0.8]},
#         ),
#         dict(
#             type="pie",
#             labels=[WELL_TYPES[i] for i in aggregate.index],
#             values=aggregate["API_WellNo"],
#             name="Well Type Breakdown",
#             hoverinfo="label+text+value+percent",
#             textinfo="label+percent+name",
#             hole=0.5,
#             marker=dict(colors=[WELL_COLORS[i] for i in aggregate.index]),
#             domain={"x": [0.55, 1], "y": [0.2, 0.8]},
#         ),
#     ]
#     layout_pie["title"] = "Production Summary: {} to {}".format(
#         year_slider[0], year_slider[1]
#     )
#     layout_pie["font"] = dict(color="#777777")
#     layout_pie["legend"] = dict(
#         font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
#     )

#     figure = dict(data=data, layout=layout_pie)
#     return figure


# # Selectors -> count graph
# @app.callback(
#     Output("count_graph", "figure"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def make_count_figure(well_statuses, well_types, year_slider):

#     layout_count = copy.deepcopy(layout)

#     dff = filter_dataframe(df, well_statuses, well_types, [1960, 2017])
#     g = dff[["API_WellNo", "Date_Well_Completed"]]
#     g.index = g["Date_Well_Completed"]
#     g = g.resample("A").count()

#     colors = []
#     for i in range(1960, 2018):
#         if i >= int(year_slider[0]) and i < int(year_slider[1]):
#             colors.append("rgb(123, 199, 255)")
#         else:
#             colors.append("rgba(123, 199, 255, 0.2)")

#     data = [
#         dict(
#             type="scatter",
#             mode="markers",
#             x=g.index,
#             y=g["API_WellNo"] / 2,
#             name="All Wells",
#             opacity=0,
#             hoverinfo="skip",
#         ),
#         dict(
#             type="bar",
#             x=g.index,
#             y=g["API_WellNo"],
#             name="All Wells",
#             marker=dict(color=colors),
#         ),
#     ]

#     layout_count["title"] = "Completed Wells/Year"
#     layout_count["dragmode"] = "select"
#     layout_count["showlegend"] = False
#     layout_count["autosize"] = True

#     figure = dict(data=data, layout=layout_count)
#     return figure


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
