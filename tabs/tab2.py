# =====================================
#     Tab 2 - Long Term Forecasts
# =====================================
from app import app, df_monthly_actual, df_monthly_modified  # Import datasets from app.py
from dash.dependencies import Input, Output, State
#from long_term_forecast_code_edited import df_monthly
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import base64
from dash.exceptions import PreventUpdate

# ========================
#      ML Model Code
# ========================
# See https://www.wintellect.com/creating-web-apps-machine-learning-models-dash/
df_monthly = pd.read_csv('data/monthly_forecast_future.csv')

# ============================================================
# Processing dataframes before plotting (Actual + Forecasted)
# ============================================================

# Getting last recorded date in the actual dataset (to be linked to forecasted graph)
last_actual_date = df_monthly_actual['MMYYYY'].max()

# Creating 2 dataframes to allow different colours for the 2 graphs
df_monthly_overlap = df_monthly[(df_monthly['TYPE'] == 'Forecasted') & (df_monthly['MMYYYY'] <= last_actual_date)]
df_monthly_future = df_monthly[(df_monthly['TYPE'] == 'Forecasted') & (df_monthly['MMYYYY'] > last_actual_date)]

# Taking first row to connect with the previous df (so that chart line is all connected, without any line breaks)
first_forecasted_row = df_monthly_future.head(1)
df_monthly_overlap = df_monthly_overlap.append(first_forecasted_row)

# ===============================
#    Model Mockup Image Setup
# ===============================
long_term_model_img = 'assets/longterm_model.png' # replace with your own image
long_term_model_encoded_img = base64.b64encode(open(long_term_model_img, 'rb').read())


# ==========================
#   Creating Tab Layout
# ==========================

layout = html.Div(

    children=[

        # ========================================
        #           Header Title
        # ========================================

        html.H2('USEP Long-Term Forecast',
            style = {'color':'white',
             'textAlign':'Center',
             'background-color': '#293796',
             'height':'28px',
             'padding':'6px',
             'margin-bottom':'-6px'}),


    # ====================
    #       Graph
    # ===================
    html.Div([
        html.Div([
            dcc.Graph(id='long-term-forecast',
                figure={
                    'data': [

                        # ====================
                        #     Actual Data
                        # ====================
                        go.Scatter(
                            line = {
                                "color": "#3E9651",
                            },
                            mode = "markers+lines",
                            name = "Actual",
                            # Plotting actual values
                            x = df_monthly_actual['MMYYYY'].unique(),
                            y = df_monthly_actual["USEP"],
                        ),

                        # ===============================
                        #  Overlap Actual and Forecasted
                        # ===============================
                        go.Scatter(
                            line = {
                                "color": "red",
                                "dash": "dot"
                            },
                            mode = "markers+lines",
                            name = "Forecast (Overlap)",
                            # Plotting data forecast values that has overlaps with actual value ground truth
                            x = df_monthly_overlap['MMYYYY'].unique(),
                            y = df_monthly_overlap["USEP"],
                        ),

                        # ===============================
                        #      Forecasted Data
                        # ===============================
                        go.Scatter(
                            line = {
                                "color": "blue",
                                "dash": "dot"  #can be changed as wanted
                            },
                            mode = "markers+lines",
                            name = "Forecast",
                            # Plotting forecasted data
                            x = df_monthly_future['MMYYYY'].unique(),
                            y = df_monthly_future["USEP"],
                        ),

                    ],
                    'layout': go.Layout(
                        yaxis=dict(
                            title="Average USEP ($/MWh)",
                            title_font = {"size": 15},
                            showgrid = False
                        ),
                        xaxis=dict(
                            title="Month-Year",
                            nticks=20,
                            title_font = {"size": 15},
                            title_standoff = 55,             # Shift axis title downwards in position
                            showgrid = False
                        ),
                        font = dict(color="#000000"),
                        hovermode='x',
                        title = '',
                        #legend = dict(orientation='h',xanchor = 'center', x = 0.5), # Reposition legend
                        )
                }
                ),
            ],
            style={"margin":"auto","padding":"0.5%"}),
        ]),



    # ==========================================
    #   Tab Header Title (Forecast Modification)
    # ==========================================

    html.Div([
        html.H2('Forecast Modification',
         style = {'color':'white',
                 'textAlign':'Center',
                 'background-color': '#293796',
                 'height':'28px',
                 'padding':'6px'})
    ]),


    # ========================================
    #   Long Term Forecast Modification Inputs
    # ========================================
    html.Div(
        children = [
                html.Div(className = 'select-title',
                style = {'font-size':'18px',
                        'verticalAlign':'top',
                        'textAlign':'Center'},
                children = [
                html.H5("Enter new inputs for forecast update")
                ]),

                # dcc.DatePickerSingle(
                #         id='date-picker-long-term',
                #         min_date_allowed=dt(2018, 1, 1),
                #         max_date_allowed=dt(2020, 12, 31),
                #         date=dt(2020,5,1),
                #         display_format='MMM YYYY',
                #         className="date-picker-long-term",
                #         with_portal=True,
                #         ),

                html.Div(style = {'textAlign':'center'},
                children = [
                html.Div(className = 'month-input',
                children = [
                dcc.Dropdown(id = 'month-input',
                    #className = 'month-input',
                    options=[{'label': 'Jan', 'value': 1}, {'label': 'Feb', 'value': 2}, {'label': 'Mar', 'value': 3},
                    {'label': 'Apr', 'value': 4}, {'label': 'May', 'value': 5}, {'label': 'Jun', 'value': 6},
                    {'label': 'Jul', 'value': 7}, {'label': 'Aug', 'value': 8}, {'label': 'Sep', 'value': 9},
                    {'label': 'Oct', 'value': 10}, {'label': 'Nov', 'value': 11}, {'label': 'Dec', 'value': 12}],
                    value='5', searchable=True, placeholder='Select Month', clearable = False,
                    style = {'width':'86px',
                            'height':'33px',
                            'verticalAlign':'bottom'}
                    )
                ]),

                html.Div(className = 'year-input',
                children = [
                dcc.Dropdown(id = 'year-input',
                    #className = 'year-input',
                    options=[{'label': i, 'value': i} for i in range(2018,2024)],
                    value='2020',
                    searchable=False,
                    clearable = False,
                    placeholder='Select Year',
                    style = {'width':'86px',
                            'height':'33px',
                            'verticalAlign':'bottom'}
                    )

                ]),

                ]),


                html.Div(style = {'text-align':'center'},
                children = [

                # Allow manual input of numeric value of brent price
                html.Div(className = 'brent-price-input',
                children = [
                html.P('Brent Price ($USD)', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'brent-price-input',
                    #className = 'brent-price-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'150px',
                            'height':'30px',
                            'verticalAlign':'middle'}
                    )
                ]),

                # Allow manual input of numeric value of exchange rate
                html.Div(className = 'exchange-rate-input',
                children = [
                html.P('Exchange Rate ($USD-$SGD)', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'exchange-rate-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'220px',
                            'height':'30px',
                            'verticalAlign':'middle'}
                    )
                ]),

                # Allow manual input of numeric value of generation capacity
                html.Div(className = 'gen-capacity-input',
                children = [
                html.P('Generation Capacity (MWh)', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'gen-capacity-input',
                    #className = 'gen-capacity-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'210px',
                            'height':'30px',
                            'verticalAlign':'middle'}
                            )
                ]),

                ]),

                html.Div(style = {'text-align':'center'},
                children = [

                # Allow manual input of numeric value of demand
                html.Div(className = 'demand-input',
                children = [
                html.P('Demand (MW)', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'demand-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'160px',
                            'height':'30px',
                            'verticalAlign':'middle'}
                            )
                ]),

                # Allow manual input of numeric value of peak demand
                html.Div(className = 'peak-demand-input',
                children = [
                html.P('Peak Demand (MW)', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'peak-demand-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'160px',
                            'height':'30px',
                            'verticalAlign':'middle',
                            'horizontalAlign':'center'}
                            )
                ]),

                # Allow manual input of numeric value of peak demand
                html.Div(className = 'production-index-input',
                children = [
                html.P('Singapore Industrial Production Index', style={"height": "auto","margin-bottom": "auto",
                                            "font-size":"13px", "textAlign":"center"}),
                dcc.Input(id = 'production-index-input',
                    type = 'number',
                    placeholder= '',
                    style = {'font-size':'14px',
                            'text-align':'center',
                            'width':'260px',
                            'height':'30px',
                            'verticalAlign':'middle'}
                            )
                ]),

                html.Div([

                html.Div(style = {'text-align':'center'},
                className = 'long-term-submit-button',
                children = [
                html.Button('Update Forecast',
                        n_clicks=0,
                        id = 'long-term-submit-button',
                        style = {'font-size':'15px',
                                'color':'white',
                                'background-color':'#6F78F3',
                                'width':'610px',
                                'height':'40px'})
                ])
                ])
            ]),
    ]),

    # ============================================
    #         Graph (Forecast Modification)
    # ===========================================
    html.Div([
        html.Div([
            dcc.Graph(id='long-term-forecast',
                figure={
                    'data': [

                        # ===============================
                        #  Overlap Actual and Forecasted
                        # ===============================

                        # Not showing the Red line on the updated forecast section
                        # go.Scatter(
                        #     line = {
                        #         "color": "red",
                        #         "dash": "dot"
                        #     },
                        #     mode = "markers+lines",
                        #     name = "Forecast (Overlap)",
                        #     # Plotting data forecast values that has overlaps with actual value ground truth
                        #     x = df_monthly_overlap['MMYYYY'].unique(),
                        #     y = df_monthly_overlap["USEP"],
                        # ),

                        # ===============================
                        #      Forecasted Data
                        # ===============================
                        go.Scatter(
                            line = {
                                "color": "blue",
                                "dash": "dot"  #can be changed as wanted
                            },
                            mode = "markers+lines",
                            name = "Forecast",
                            # Plotting forecasted data
                            x = df_monthly_future['MMYYYY'].unique(),
                            y = df_monthly_future["USEP"],
                        ),

                    ],
                    'layout': go.Layout(
                        yaxis=dict(
                            title="Average USEP ($/MWh)",
                            title_font = {"size": 15},
                            showgrid = False
                        ),
                        xaxis=dict(
                            title="Month-Year",
                            nticks=20,
                            title_font = {"size": 15},
                            title_standoff = 55,             # Shift axis title downwards in position
                            showgrid = False
                        ),
                        font = dict(color="#000000"),
                        hovermode='x',
                        title = 'Updated Long Term Forecast',
                        #legend = dict(orientation='h',xanchor = 'center', x = 0.5), # Reposition legend
                        )
                }
                ),
            ],
            style={"margin":"auto","padding":"0.5%"}),
        ]),


    # # ====================
    # #  Model Details Header
    # # ===================
    #     html.Div([html.H2('Details of Long-Term Forecast Model',
    #          style = {'color':'white',
    #                  'textAlign':'Center',
    #                  'background-color': '#003D7C',
    #                  'height':'28px',
    #                  'padding':'6px'})
    #                  ]),
    #
    # # ====================
    # #  Model Mockup Image
    # # ===================
    #
    #     html.Img(src='data:image/png;base64,{}'.format(long_term_model_encoded_img.decode()),
    #             className="model-details")

])


# =======================================================
#   Callback for Long Term Forecast Modification (Tab 2)
# =======================================================
# Linking to a single fixed dataset, as this is for demo purpose only
# i.e. Display a single new graph upon clicking Update Forecast button

@app.callback(
    Output('long-term-forecast', 'figure'),
    [Input('long-term-submit-button', 'n_clicks')],
    [State('production-index-input', 'value')])
def update_long_term_forecast_modified(n_clicks, number):

    if n_clicks > 0:
        placeholder = number
        traces = []

        # Not showing the Red line on the updated forecast section
        # traces.append(
        #     go.Scatter(
        #         line = {
        #             "color": "red",
        #             "dash": "dot"
        #         },
        #         mode = "markers+lines",
        #         name = "Forecast (Overlap)",
        #         # Plotting data forecast values that has overlaps with actual value ground truth
        #         x = df_monthly_overlap['MMYYYY'].unique(),
        #         y = df_monthly_overlap["USEP"],
        #     ))

        traces.append(
            go.Scatter(
                line = {
                    "color": "blue",
                    "dash": "dot"  #can be changed as wanted
                },
                mode = "markers+lines",
                name = "Forecast",
                # Plotting forecasted data
                x = df_monthly_modified['MMYYYY'],
                y = df_monthly_modified["USEP_m"],
            ))

        return {
            'data' : traces,
            'layout': go.Layout(
                yaxis=dict(
                    title="Average USEP ($/MWh)",
                    title_font = {"size": 15},
                    showgrid = False
                    ),
                    xaxis=dict(
                    title="Month-Year",
                    nticks=20,
                    title_font = {"size": 15},
                    title_standoff = 55,             # Shift axis title downwards in position
                    showgrid = False
                    ),
                    font = dict(color="#000000"),
                    hovermode='x',
                    title = 'USEP Mid to Long Term Forecast',
                    #legend = dict(orientation='h',xanchor = 'center', x = 0.5), # Reposition legend
                    )
                    }

    else:
        raise PreventUpdate  # This ensures that the chart is in the original form, unless we click the Submit button
