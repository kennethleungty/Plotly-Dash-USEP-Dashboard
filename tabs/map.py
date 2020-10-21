import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from app import app, server
from plotly import graph_objs as go
from plotly.graph_objs import *
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from datetime import date, timedelta
import re

mapbox_access_token = 'pk.eyJ1Ijoia2x0eTA5ODgiLCJhIjoiY2s5ODlsbTk2MDczNTNncGxpMGlhcHBzaSJ9.nt4XmPRONhG8oUoiraS6vw'

df_map_data = pd.read_csv('data/map_mock_data.csv')
df_map_data['Date'] = pd.to_datetime(df_map_data['Date'], dayfirst = True)
df_map_data['Date'] = pd.DatetimeIndex(df_map_data['Date']).strftime("%Y-%m-%d")

# Creating a list of the genco names
list_of_names = list(df_map_data['Name'].unique())
list_of_names.sort()

layout = html.Div(
        children = [


    # =====================
    #   Tab Header Title
    # =====================

    # html.Div([
    #     html.H2('Upcoming Advisories',
    #      style = {'color':'white',
    #              'textAlign':'Center',
    #              'background-color': '#293796',
    #              'height':'28px',
    #              'padding':'6px'})
    # ]),

    # =====================
    #      Date Picker
    # =====================

    html.Div([
        html.H4("Displaying week-ahead advisories from: ",
        className="map-date-picker-title",
        id="map-date-picker-title"),

        dcc.DatePickerSingle(
                id='map-date-picker',
                min_date_allowed=dt(2019, 1, 1),
                max_date_allowed=dt(2020, 12, 31),
                date=str(dt(2019, 12, 25)),   # Default value (using 25th Dec 2019 as demo)
                display_format='DD MMM YYYY',
                className="map-date-picker",
                #with_portal=True   # When True, the date picker popup will appear in the center of the screen for the user to pick
                ),


    html.Div([

    # =====================================
    #   Dropdown Options (Selecting Genco)
    # =====================================

    html.Div([
        dcc.Dropdown(
            id = 'map-location-dropdown',
            className = 'map-location-dropdown',
            options=[{"label": i, "value": i} for i in list_of_names],
            #options=[{"label": i, "value": j} for i,j in zip(list_of_names, list_of_symbols)],
            value = list_of_names,
            multi = True,
            clearable = True,
            style = {'font-size':'16px'}
            ),
            ]),

    # ============================
    #        Geospatial Map
    # ============================

    html.Div(className = 'map-graph',
        children = [
            dcc.Graph(id = 'map-graph',
                    style = {'width':'910px',
                            'margin-top':'-20px'})
        ]
    ),

        ])

    ]),

    # =====================================
    #   Checklist Options (Selecting Genco)
    # =====================================

    # html.Div([
    #     dcc.Checklist(
    #         id = 'map-location',
    #         className = 'map-location',
    #         options=[{"label": i, "value": i} for i in list_of_names],
    #         #options=[{"label": i, "value": j} for i,j in zip(list_of_names, list_of_symbols)],
    #         value = list_of_names,
    #         labelStyle = {'display': 'inline-block', 'cursor': 'pointer', "vertical-align":"middle"},
    #         inputStyle = {'margin-left':'6px', 'margin-right':'6px'},
    #         style={"display":"inline-flex", "flex-wrap":"wrap", "justify-content":"space-between","line-height":"32px"},
    #         )
    #     ]),



    # ============================
    #       Map Data Table
    # ============================

    html.Div([
        dash_table.DataTable(
            id = 'map-data-table',
            data = df_map_data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_map_data.columns],
            #row_selectable= 'multi',
            #selected_rows=[],
            hidden_columns=['Latitude', 'Longitude'],
            style_header={'backgroundColor': '#DEE5FF',
                'fontWeight': 'bold',
                'textAlign': 'Center'},
            style_data = {                      # Allowing text wrapping (Overflow into multiple lines)
                    'whiteSpace': 'normal',
                    'height': 'auto'},
            style_cell={'padding': '3px',
                'textAlign': 'Center',
                'font-family':'sans-serif',
                'font-size':'12.5px'},
            style_data_conditional=[{             # Alternating row background colours
            'if': {'row_index': 'odd'},
                    'backgroundColor': '#F2F3F4'
                    }],
        style_as_list_view=True,
        css= [{"selector": ".show-hide", "rule": "display: none"}]
        )
    ])

    ])


# =====================================
#     Functions for Map Generation
# =====================================
layout_map = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#FFFCFC',
    paper_bgcolor='white',
    legend=dict(font=dict(size=10), orientation='h'),
    title='',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lat=1.359291,
            lon=103.809529
        ),
        zoom=10,
    )
)


def gen_map(map_data):
    # groupby returns a dictionary mapping the values of the first field
    # 'classification' onto a list of record dictionaries with that
    # classification value.
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(map_data['Latitude']),
                "lon": list(map_data['Longitude']),
                "hoverinfo": "text",
                "hovertext": [["Name: {} <br>Date: {} <br>Condition: {}".format(i,j,k)]
                                for i,j,k in zip(map_data['Name'],
                                                 map_data['Date'],
                                                 map_data['Condition'])],
                "mode": "markers",
                "name": list(map_data['Name']),
                "marker": {
                    "size": 6,
                    "opacity": 0.7
                }
        }],
        "layout": layout_map
    }



# =======================================
#         Callback for Map
# =======================================

@app.callback(
    Output('map-graph', 'figure'),
    [Input('map-date-picker', 'date'),
    Input('map-location-dropdown', 'value')])

def update_map(date, locations):
    new_df = df_map_data.copy()
    new_df = new_df.loc[(new_df['Date'] >= date) &
                    (new_df['Name'].isin(locations))]
    return gen_map(new_df)

# def generate_map(locations):
#     fig = go.Figure(go.Scattermapbox(
#             lat=['1.359291'],
#             lon=['103.809529'],
#             mode='markers',
#             marker=go.scattermapbox.Marker(
#             size=14
#             ),
#             text=['Singapore'],
#             ))
#
#     fig.update_layout(
#             hovermode='closest',
#             mapbox=dict(
#             accesstoken=mapbox_access_token,
#             bearing=0,
#             center=go.layout.mapbox.Center(
#             lat=1.359291,
#             lon=103.809529
#             ),
#             pitch=0,
#             zoom=7
#                 )
#         )
#
#     return fig


# =======================================
#      Callback for Map Data Table
# =======================================
@app.callback(
    Output('map-data-table', 'data'),
    [Input('map-date-picker', 'date'),
    Input('map-location-dropdown', 'value')])

def update_map_data_table(date, locations):
    df_map_data_modi = df_map_data.copy()

    # Strptime will allow us to use comparators (> < =) on the date
    new_date = dt.strptime(date.split(' ')[0], '%Y-%m-%d')

    # Convert to datetime
    df_map_data_modi['Date'] = pd.to_datetime(df_map_data_modi['Date'], format='%Y-%m-%d')

    week_ahead_date = new_date + timedelta(days=7)
    df_map_data_modi = df_map_data_modi.loc[(df_map_data_modi['Date'] >= new_date) &
                    (df_map_data_modi['Date'] < week_ahead_date) &
                    (df_map_data_modi['Name'].isin(locations))]
    df_map_data_modi['Date'] = df_map_data_modi['Date'].dt.date  # This removes the trailing T00:00:00 in the date column

    return df_map_data_modi.to_dict("rows")
