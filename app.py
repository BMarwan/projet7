# Import required libraries
import pathlib
import dash
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import dash_table
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


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Load data

test_corrs_removed = pd.read_csv(DATA_PATH.joinpath("test_bureau_corrs_removed.csv"), low_memory=False)

test_features = pd.read_csv(DATA_PATH.joinpath("test_features.csv"), low_memory=False)
test_features.drop('Unnamed: 0', axis=1, inplace=True)
test_features = test_features.reindex(sorted(test_features.columns), axis=1)


# test_corrs_removed = test_corrs_removed.sample(n=10, random_state=1)
# test_corrs_removed = test_corrs_removed.sort_values('SK_ID_CURR')

client_id = test_corrs_removed["SK_ID_CURR"].tolist()

#Load models

clf_non_solvable = load(DATA_PATH.joinpath('lgb_pos_weights.joblib'))
clf_solvable = load(DATA_PATH.joinpath('lgb_neg_weights.joblib'))

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

PAGE_SIZE = 5

def calcul_interpretation(clf, client_id):

    test_features_filled= test_features.fillna(test_features.median())

    lime1 = LimeTabularExplainer(test_features_filled,
                                feature_names=test_features_filled.columns,
                                discretize_continuous=False)
                                
    explain_data = test_features_filled.iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)]].T.squeeze()
    exp = lime1.explain_instance(explain_data,
                                clf.predict_proba,
                                num_samples=1000)

    exp_list = exp.as_list()
    exp_keys = []
    exp_values = []
    exp_positives = []
    for i in range(len(exp_list)):
        exp_keys.append(exp_list[i][0])
        exp_values.append(exp_list[i][1])
    

    df_data = pd.DataFrame(data=[exp_keys,exp_values])
    df_data = df_data.T
    df_data.columns=['exp_keys', 'exp_values']
    df_data = df_data.iloc[np.abs(df_data['exp_values'].values).argsort()]
    df_data['color'] = df_data.exp_values.apply(lambda x: 'red' if x > 0 else 'green')

    return df_data

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
                            src=app.get_asset_url("assets/logo.png"),
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
                            id="model_selector",
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
                            # [
                            # dash_table.DataTable(
                            #     id='table',
                            #     columns=[{"name": i, "id": i} for i in clients_similaires.columns],
                            #     page_current=0,
                            #     page_size=PAGE_SIZE,
                            #     page_action='custom'
                            # )
                            # ],#dcc.Graph(id="count_graph")
                            id="countGraphContainer",
                            className="pretty_container",
                            style={"width":"100%"},
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
                    [   html.H6('Comparaison d\'un client avec la totalité des clients', style={'text-align':'center'}),
                        dcc.Dropdown(
                            id="situation",
                            options=[{'label': i, 'value': i} for i in test_features.columns.values.tolist()],
                            multi=False,
                            value='AMT_INCOME_TOTAL',
                            className="dcc_control",
                        ),
                        dcc.Graph(id="client_situation")
                    
                    ],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
    
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
    [Input("model_selector", "value"), 
    Input('client_id', 'value')]
)
def display_status(selector, client_id):
    if selector == "solvable":
        clf = clf_solvable

    elif selector == "nonsolvable":
        clf = clf_non_solvable
         
    defaut_precentage = clf.predict_proba(test_features.iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)]].values.reshape(1, -1))[0][1]
    return '{} %'.format(int(np.round(defaut_precentage*100, decimals=0)))


@app.callback(
    Output('main_graph', 'figure'),
    [Input('client_id', 'value'),
    Input("model_selector", "value")]
)

def update_graph(client_id, selector):

    if selector == "solvable":
        clf = clf_solvable

    elif selector == "nonsolvable":
        clf = clf_non_solvable

    df_data = calcul_interpretation(clf, client_id)

    fig = go.Figure(go.Bar(
                x=df_data['exp_values'],
                y=df_data['exp_keys'],
                orientation='h',
                marker_color=df_data['color'])            
                )

  
    return fig

@app.callback(
    Output('countGraphContainer', 'children'),
    [Input('client_id', 'value'),
     Input("model_selector", "value")]
)

def client_similarities(client_id, selector):

    if selector == "solvable":
        clf = clf_solvable

    elif selector == "nonsolvable":
        clf = clf_non_solvable

    test_features_filled = test_features.fillna(test_features.median())

    #Train the algo
    neigh = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(test_features_filled)
    
    index = test_corrs_removed[test_corrs_removed["SK_ID_CURR"] == int(client_id)].index.values

    data_client = test_features_filled.iloc[index[0]]
    data_client = data_client.values.reshape(1, -1)

    #Find the 5 nearest neighborhoods
    indices = neigh.kneighbors(data_client, return_distance=False)
    
    clients_similaires = pd.DataFrame()
    client_id_similaire = pd.DataFrame()

    
    #Interprétation lime
    df_data = calcul_interpretation(clf, client_id)

    #Extract informations of the 5 nearest
    for i in indices:
        clients_similaires = clients_similaires.append(test_features_filled.iloc[i])
        client_id_similaire = client_id_similaire.append(test_corrs_removed["SK_ID_CURR"].iloc[i])
    
    client_id_similaire = client_id_similaire.T
    client_id_similaire.columns=["SK_ID_CURR"]
    client_id_similaire["SK_ID_CURR"].astype(int)

    clients_similaires = clients_similaires[df_data['exp_keys']]

    df_result = pd.concat([client_id_similaire, clients_similaires], axis=1)

    return [
                html.H6('Comparaison avec les 4 clients les plus ressemblants', style={'text-align':'center'}),
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in df_result.columns],
                    data=df_result.to_dict("rows"),
                    # page_current=0,
                    # page_size=PAGE_SIZE,
                    # page_action='custom',
                    style_table={'overflowX': 'auto'},
                    
                )
    ]



@app.callback(
    Output('client_situation', 'figure'),
    [Input('client_id', 'value'),
    Input("situation", "value")]
)

def update_graph_situation(client_id, situation):

    feature = test_features[situation].iloc[test_corrs_removed.index[test_corrs_removed['SK_ID_CURR']==int(client_id)][0]]
    moyenne = test_features[situation].mean()
    mediane = test_features[situation].median()

    client_situation = pd.DataFrame(data=[feature,moyenne,mediane], columns=['Infos'])
    client_situation.index=['Client', 'Moyenne', 'Mediane']

    fig = go.Figure([go.Bar(x=client_situation.index, y=client_situation['Infos'])])

  
    return fig


# Main
if __name__ == "__main__":
    app.run_server(debug=False)
