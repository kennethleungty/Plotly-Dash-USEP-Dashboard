# =====================================
#    Tab 1 - Day Ahead Forecasts
# =====================================
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import dash_table
from app import df_dayahead, df_advisory, df_alerts
import plotly.graph_objs as go
import base64
import numpy as np
import pandas as pd

# # ==========================
# #  Model Mockup Image Setup
# # ==========================
# day_ahead_model_img = 'assets/shortterm_model.png' # replace with your own image
# day_ahead_model_encoded_img = base64.b64encode(open(day_ahead_model_img, 'rb').read())

# ==========================
#   Creating Tab Layout
# ==========================

layout = html.Div(
    children=[

    # =====================
    #   Tab Header Title
    # =====================

    html.Div([
        html.H2('USEP Day-Ahead Forecast',
         style = {'color':'white',
                 'textAlign':'Center',
                 'background-color': '#293796',
                 'height':'28px',
                 'padding':'6px'})
    ]),


    # =====================
    #      Date Picker
    # =====================

    html.Div([
        html.H3("Select date: ",
        className="date-picker-title",
        id="date-picker-title"),

        dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=dt(2018, 1, 1),
                max_date_allowed=dt(2020, 12, 31),
                date=str(dt(2020,1,15)), #Use 15 Jan 2020 as starting date for demo
                #date=str(df_dayahead["DATE"].max()),   # Default value
                display_format='DD MMM YYYY',
                className="date-picker",
                with_portal=True,   # When True, the date picker popup will appear in the center of the screen for the user to pick
                ),

        html.Div(id='output-container-date-picker',   # Container that will output the Display Information for (date) text
        className = 'output-container-date-picker')
    ]),


    # ================================
    #   USEP Line Graph (Day ahead)
    # ================================
    html.Div([
        dcc.Graph(id='graph-usep-day-ahead')
    ]),


    # ====================
    #   Advisory Header
    # ====================

    html.Div('Advisory',
    id='advisory-date',
    className = 'advisory-date'),

    # ====================
    #    Alert Header
    # ====================

    html.Div('Alerts',
    id='alert-date',
    className = 'alert-date'),

    # ====================
    #   Advisory Table
    # ====================

    html.Div(className = 'advisory-table',
        children = [

        dash_table.DataTable(
        id='advisory-table',
        columns=[{"name": "Date", "id": "Date"},
                {"name": "From", "id": "From time"},
                {"name": "To", "id": "To time"},
                {"name": "Condition", "id": "Condition"}],
        hidden_columns=['Date'],            # Hide the date column from display
        style_data = {                      # Allowing text wrapping (Overflow into multiple lines)
                'whiteSpace': 'normal',
                'height': 'auto'},
        css=[{"selector": ".show-hide", "rule": "display: none"}, # Hide toggle button
             {'selector': 'tr:hover', 'rule': 'background-color: pink;'}],
        style_header={'backgroundColor': '#DEE5FF',
                    'fontWeight': 'bold',
                    'textAlign': 'Center'},
        style_cell={'padding': '5px',
                    'textAlign': 'Center',
                    'font-family':'sans-serif',
                    'font-size':'12px'},
        style_as_list_view=True,              # Removes borders of table
        style_data_conditional=[{             # Alternating row background colours
            'if': {'row_index': 'odd'},
            'backgroundColor': '#F2F3F4'
            }]
        )
    ]),


    # ====================
    #     Alert Table
    # ====================

    html.Div(className = 'alert-table',
        children = [

        dash_table.DataTable(
        id='alert-table',
        columns=[{"name": i, "id": i} for i in df_alerts.columns],
        hidden_columns=['Date'],            # Hide the date column from display
        style_data = {                      # Allowing text wrapping (Overflow into multiple lines)
                'whiteSpace': 'normal',
                'height': 'auto'},
        css=[{"selector": ".show-hide", "rule": "display: none"}, # Hide toggle button
             {'selector': 'tr:hover', 'rule': 'background-color: pink;'}],
        style_header={'backgroundColor': '#F0CCD6',
                    'fontWeight': 'bold',
                    'textAlign': 'Center'},
        style_cell={'padding': '5px',
                    'textAlign': 'Center',
                    'font-family':'sans-serif',
                    'font-size':'13px'},
        style_data_conditional=[{             # Alternating row background colours
            'if': {'row_index': 'odd'},
            'backgroundColor': '#F2F3F4'
            }]
        )
    ]),



    # ==========================================
    #   Tab Header Title (Forecast Modification)
    # ==========================================

    html.Div([
        html.H2('Updated and User-Defined Forecasts',
         style = {'color':'white',
                 'textAlign':'Center',
                 'background-color': '#293796',
                 'height':'28px',
                 'padding':'6px',
                 'margin-bottom':'-2px'})
    ]),

    # ===============================
    #   Day-Ahead Chart Modification
    # ===============================
    html.Div([
        dcc.Graph(id='graph-modi-usep-day-ahead')    # Actual graph object
    ]),

# Reset button: https://community.plotly.com/t/how-to-implement-reset-button-to-clear-output/15701/2

    # =================================
    #   Day-Ahead Modifications Inputs
    # =================================

    html.Div(style = {'text-align':'center'},
        children = [

            dcc.Dropdown(id = 'period-selector',
            className = 'period-selector',
            options=[{'label': f'Period {i}', 'value': i} for i in range(1,49,1)], # Allow user to pick Period 1 - Period 48
            #options=[{'label': 'All Periods', 'value': '100'}],
            value=[i for i in range (1,49,1)],   # Default is that all Periods are picked at the start
            multi=True,                          # Allows for multiple options to be selected
            searchable=False,
            placeholder='Select Period(s) to Modify'         # Placeholder text to display
            ),

            dcc.Dropdown(id = 'scale-selector',
                    className = 'scale-selector',
                    options=[{'label': f'Scale selected periods by {i}', 'value': i} for i in np.arange(2.0,0.0,-0.1).round(1)],
                    value='',           # No default value set
                    searchable=False,
                    placeholder='Select Scaling Factor'
                    ),

            html.Button('Modify Forecast', id='day-ahead-submit-button',
                            style = {'font-size':'20px',
                                    'color':'white',
                                    'background-color':'#6F78F3'},
                            className = 'day-ahead-submit-button'),

    ]),

    # ==========================================
    #           Header Title
    # ==========================================

    html.Div([
        html.H2('Forecasts for Previous Day and Following Day',
         style = {'color':'white',
                 'textAlign':'Center',
                 'background-color': '#293796',
                 'height':'28px',
                 'padding':'6px',
                 'margin-bottom':'-1px'})
    ]),

        html.Div(style = {'textAlign':'center'},
        children = [

        html.Div(id='header-previous-day',   # Container that will output the Display Information for (date) text
        className = 'header-previous-day'),

        html.Div(id='header-following-day',   # Container that will output the Display Information for (date) text
        className = 'header-following-day')

        ]),


    # ==================================
    #      Previous Day Forecast
    # ==================================

    html.Div(className = 'graph-prev-day',
    children = [
        # html.H2('Previous Day',
        #  style = {'color':'white',
        #          'textAlign':'Center',
        #          'background-color': 'green',
        #          'height':'28px',
        #          'padding':'6px'}),

        dcc.Graph(id='graph-prev-day')
    ]),


    # ==================================
    #     Following Day Forecast
    # ==================================

    html.Div(className = 'graph-following-day',
    children = [
    #     html.H2('Following Day',
    #      style = {'color':'white',
    #              'textAlign':'Center',
    #              'background-color': 'red',
    #              'height':'28px',
    #              'padding':'6px'}),
    #
        dcc.Graph(id='graph-following-day')
    ]),


    # ========================
    #   Model Details Header
    # ========================

    # html.Div([html.H2('Details of Day-Ahead Forecast Model',
    #  style = {'color':'white',
    #          'textAlign':'Center',
    #          'background-color': '#003D7C',
    #          'height':'28px',
    #          'padding':'6px'})
    #          ]),
    #
    # # ==========================
    # #    Model Mockup Image
    # # ==========================
    #
    # html.Img(src='data:image/png;base64,{}'.format(day_ahead_model_encoded_img.decode()),
    #     className="model-details")

    ])
