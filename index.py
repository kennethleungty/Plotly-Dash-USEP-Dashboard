# ========= Dashboard  ===========
#      Main Layout (Index.py)
# ================================
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import dash_table
import calendar
import re
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from datetime import date, timedelta
from weather import min_temp, max_temp, condition   # Importing weather data retrieved from NEA API

# Importing dataframes from app.py, and also the 3 tab layouts
from app import app, server, df_dayahead, df_advisory, df_alerts, df_actual
from tabs import tab1, tab2, tab3, tab4, map

# Multi-app deployment error: https://community.plotly.com/t/nolayoutexception-on-deployment-of-multi-page-dash-app-example-code/12463/2

# ======================================
#    Data Pre-Processing for Datasets
# ======================================

# Creating column for period time (for the 48 periods)
list_of_periods = ['00:00','00:30','01:00','01:30','02:00','02:30','03:00','03:30',
                   '04:00','04:30','05:00','05:30','06:00','06:30','07:00','07:30',
                   '08:00','08:30','09:00','09:30','10:00','10:30','11:00','11:30',
                   '12:00','12:30','13:00','13:30','14:00','14:30','15:00','15:30',
                   '16:00','16:30','17:00','17:30','18:00','18:30','19:00','19:30',
                   '20:00','20:30','21:00','21:30','22:00','22:30','23:00','23:30'
                  ]
# Creating a new column based on the mapped period times above
df_dayahead['PERIOD_TIME'] = np.tile(list_of_periods, len(df_dayahead)//len(list_of_periods) + 1)[:len(df_dayahead)]
df_actual['PERIOD_TIME'] = np.tile(list_of_periods, len(df_actual)//len(list_of_periods) + 1)[:len(df_actual)]

# ======================================================
#        Data Pre-Processing for Welcome Intro Text
# ======================================================
# Note: Enter this in cmd to change Heroku app timezone to Singapore time: heroku config:add TZ="Asia/Singapore" --app msba-emc

today = date.today()
today = today.strftime("X%d %b %Y").replace('X0','X').replace('X','') # Remove leading zero for the day component

# Creating separate today variable for extracting part of day (morning, afternoon, evening)
today_2 = date.today()
hour = dt.now().hour
day_of_week = calendar.day_name[today_2.weekday()]

def get_part_of_day(hr):
    if 0 <= hr < 12:
        return "morning"
    elif 12 <= hr < 17:
        return "afternoon"
    else:
        return "evening"

part_of_day = get_part_of_day(hour)

# Using a fake today date to populate speech bubble with forecast values first
fake_today = '13/12/2019'
df_today = df_dayahead[df_dayahead['DATE'] == fake_today]
df_advisory_today = df_advisory[df_advisory['Date'] == fake_today]

# Eventually when pipeline is ready for use of actual day, then it will be:
# df_today = df_dayahead[df_dayahead['DATE'] == date.today()]

# Obtaining rows containing the max and min USEP prices for the selected date
row_usep_max = df_today.loc[df_today['USEP'].idxmax()]
row_usep_min = df_today.loc[df_today['USEP'].idxmin()]

# Obtaining actual max and min USEP values for the selected date, along with corresponding period time
today_usep_max = row_usep_max['USEP']
today_usep_min = row_usep_min['USEP']

today_usep_max_period = row_usep_max['PERIOD_TIME']
today_usep_min_period = row_usep_min['PERIOD_TIME']

# Extracting how many advisories are active for the selected date
num_of_advisories = len(df_advisory_today.index)

# ===================================
#   Creating Main Dashboard Layout
# ===================================

app.layout = html.Div(style={'font-family':'Helvetica', 'textAlign':'justify'},

    children=[

    # =====================================================
    #    NUS and EMC Logos, and Welcome Intro paragraph
    # =====================================================
    html.Div([

    html.Img(id='nus_logo',
                 src = 'http://www.nus.edu.sg/images/default-source/base/logo.png',
                 style={'height':'13%',
                        'width':'13%'},
                        className="nus-logo"),

    html.Img(id='emc_logo',
             src = 'https://www.ddynamics.net/wp-content/gallery/Home-Gallery/EMC.png',
             style={'height':'21%',
                    'width':'21%'},
                    className="emc-logo"),

    html.Img(id='chatbot_img',
            src = 'https://i.ibb.co/wJdX0RV/chatbot.png',
            style={'height':'6%',
                    'width':'6%'},
            className="chatbot_img"),

    dcc.Markdown(f'''
    ## Good {part_of_day} boss! Here are today's highlights:

    Today is {today} ({day_of_week}). Peak forecasted USEP value is
    **{today_usep_max}**, occurring at {today_usep_max_period}h, whereas trough forecasted USEP value is
    **{today_usep_min}**, occurring at {today_usep_min_period}h. Temperature will reach a high of {max_temp}{chr(176)}C
    and a low of {min_temp}{chr(176)}C, and {condition} are expected. There are {num_of_advisories} advisories
    for you to take note of. Have an electrifying {day_of_week} {part_of_day}!
        ''',
    className = 'welcome-text', id='welcome-text')

    ]),


    # =====================
    #         Tabs
    # =====================
    html.P(            # This html.P is to create white spacing from the blocks above
     html.P(dcc.Tabs(id="tabs", value='tab1', children=[
         dcc.Tab(label='Day-Ahead Forecast', value='tab1',
            className = 'main_tab'),
         dcc.Tab(label='Long-Term Forecast', value='tab2',
            className = 'main_tab'),
         dcc.Tab(label='Market Indicators', value='tab3',
            className = 'main_tab'),
        # dcc.Tab(label='Upcoming Advisories', value='map',
        #     className = 'main_tab'),
        # dcc.Tab(label='Historical Data', value='tab4',
        #     className = 'main_tab'),
     ],colors={
        "border": "white",
        "primary": "gold",
        "background": "#F8F9F9"
    }))
        ),
     html.Div(id='tabs-content'),


    # ========================================
    #            Bottom Credits
    # ========================================
    html.Hr(),   # Adding a divider

    # html.P(html.Div(html.H4("Brought to you by Team E=MC2 (NUS MSBA 2019/2020)"),
    #          className = 'credits',
    #          style={'textAlign': "Center",
    #                 'font':'Calibri',
    #                 'background-color':'white',
    #                 'color':'black'
    #                 }
    #                 ))

])

# ==============================================================================
#                                  Callbacks
# ==============================================================================

# =======================================
#       Callback for Main Tab content
# =======================================

# Returns the respective tab based on the tab clicked

@app.callback(Output('tabs-content', 'children'),
    [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab1':
        return tab1.layout

    elif tab == 'tab2':
        return tab2.layout

    elif tab == 'tab3':
        return tab3.layout

    elif tab == 'map':
        return map.layout

    # Hiding tab 4 (Historical Data)
    # elif tab == 'tab4':
    #     return tab4.layout


# =======================================
#      Callback for Date picker Text
# =======================================

# Returns the sentence that shows what date we are displaying the information for

@app.callback(
    Output('output-container-date-picker', 'children'),
    [Input('date-picker', 'date')])
def update_output(date):
    string_prefix = 'Displaying information for '
    if date is not None:
        date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
        date_string = date.strftime('%d %b %Y')
        return string_prefix + date_string


# ==============================================
#    Callback for Day Ahead Actual + Forecast
# =============================================

# Returns the actual and forecasted USEP graphs, based on date selected from date picker

@app.callback(
    Output('graph-usep-day-ahead', 'figure'),
    [Input('date-picker', 'date')])
def update_output(selected_date):
    traces = []          # Traces is a list to contain data points
    df_filtered = df_dayahead[df_dayahead["DATE"] == selected_date]
    df_filtered_actual = df_actual[df_actual["DATE"] == selected_date]

    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered_actual['USEP'],
            mode = 'markers+lines',
            name = 'Actual',
            line = {'color': '#3E9651'}
            ))

    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered['Forecast_Baseline'],
            mode = 'markers+lines',
            name = 'Forecast',
            line = {'color': 'red',
                    'dash':'dot'}
            ))

    return {
        'data' : traces,
        'layout': go.Layout(
            title = '',
            xaxis = {'title': 'Period (hh:mm)', 'showgrid':False, 'nticks':12}, # Hide gridlines, and show 48 x ticks
            yaxis = {'title': 'USEP ($/MWh)', 'showgrid':False},
            hovermode='x')
    }

# ====================================================
#     Callback for Day Ahead Forecast Modifications
# ====================================================

# Returns a different modified forecast graph based on the date, periods, and scale factor chosen

@app.callback(
    Output('graph-modi-usep-day-ahead', 'figure'),
    [Input('date-picker', 'date'),
     Input('day-ahead-submit-button', 'n_clicks')],
    [State('scale-selector','value'),
    State('period-selector','value')])

def modify_day_ahead(selected_date, n_clicks, scale_no, periods):
    traces = []
    scale_no = pd.to_numeric(scale_no)  # Convert scaling factor to numeric (range 0.1 to 2.0)

    # First filter by selected date.
    # df_filtered is for original forecast, while df_modified is for modified graph (which will be overlapped with original forecast)
    df_filtered = df_dayahead[df_dayahead["DATE"] == selected_date]
    df_modified = df_filtered.copy()

    # Selecting periods to modify
    df_modified_change = df_modified[df_modified['PERIOD'].isin(periods)]

    # Implement scaling factor for specific periods
    df_modified_change['Forecast_Baseline'] = df_modified_change['Forecast_Baseline'].multiply(scale_no)

    # Merge the modified data points with the original unchanged data points to form back one single graph
    df_modified_nochange = df_modified[~df_modified['PERIOD'].isin(periods)]
    df_modified = pd.concat([df_modified_change, df_modified_nochange], axis = 0)

    # Graph for original forecast
    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered['Forecast_Baseline'],
            mode = 'markers+lines',
            name = 'Baseline Forecast',
            line = {'color': 'red',
                    'dash':'dot'}
            ))

    # Graph for updated model forecast
    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_modified['Updated_Model'],
            mode = 'markers+lines',
            name = 'Updated Forecast',
            line = {'color': 'purple',
                    'dash':'dot'}
            ))

    # Graph for modified forecast (user defined scaling)
    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_modified['Forecast_Baseline'],
            mode = 'markers+lines',
            name = 'User-Defined Forecast',
            line = {'color': 'orange',
                    'dash':'dot'}
            ))



    return {
        'data' : traces,
        'layout': go.Layout(
            title = '',
            xaxis = {'title': 'Period (hh:mm)', 'showgrid':False, 'nticks':12}, # Hide gridlines
            yaxis = {'title': 'USEP ($/MWh)', 'showgrid':False},
            hovermode='x')
    }


# ============================================
#   Callback for Advisory Title Header
# ============================================

# @app.callback(
#     Output('advisory-date', 'children'),
#     [Input('date-picker', 'date')])
# def update_output(date):
#     string_prefix = 'Advisory Notices for '
#     if date is not None:
#         date = dt.strptime(date.split(' ')[0], '%Y-%m-%d')
#         date_string = date.strftime('%d %b %Y')
#         return string_prefix + date_string

# ============================================
#   Callback for Alert Title Header
# ============================================

# @app.callback(
#     Output('alert-date', 'children'),
#     [Input('date-picker', 'date')])
# def update_output(date):
#     string_prefix = 'Alerts for '
#     if date is not None:
#         date = dt.strptime(date.split(' ')[0], '%Y-%m-%d')
#         date_string = date.strftime('%d %b %Y')
#         return string_prefix + date_string


# ============================================
#     Callback for Advisory Table
# ============================================

# Returns the advisory notices (dataframe rows) for the selected date

@app.callback(
    Output('advisory-table', 'data'),
    [Input('date-picker', 'date')])
def update_advisory_table(date):
    df_advisory_new = df_advisory.loc[(df_advisory['Date'] == date)]
    return df_advisory_new.to_dict("rows")


# ============================================
#     Callback for Alert Table
# ============================================

# Returns the alert values for the selected date

@app.callback(
    Output('alert-table', 'data'),
    [Input('date-picker', 'date')])
def update_alert_table(date):
    df_alerts_new = df_alerts.loc[(df_alerts['Date'] == date)]
    return df_alerts_new.to_dict("rows")


# ===========================================================
#    Callback for Headers of Prev and Following day charts
# ===========================================================
@app.callback(
    Output('header-previous-day', 'children'),
    [Input('date-picker', 'date')])
def update_output(date):
    string_prefix = 'Previous Day - '
    if date is not None:
        date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
        previous_date = date - timedelta(days=1)
        date_string = previous_date.strftime('%d %b %Y')
        return string_prefix + date_string


@app.callback(
    Output('header-following-day', 'children'),
    [Input('date-picker', 'date')])
def update_output(date):
    string_prefix = 'Following Day - '
    if date is not None:
        date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
        following_date = date + timedelta(days=1)
        date_string = following_date.strftime('%d %b %Y')
        return string_prefix + date_string


# =============================================
#    Callback for Previous Day Chart
# =============================================

@app.callback(
    Output('graph-prev-day', 'figure'),
    [Input('date-picker', 'date')])
def update_output(selected_date):
    traces = []          # Traces is a list to contain data points

    new_selected_date = dt.strptime(selected_date.split(' ')[0], '%Y-%m-%d')
    previous_date = new_selected_date - timedelta(days=1)
    df_filtered = df_dayahead[df_dayahead["DATE"] == previous_date]
    df_filtered_actual = df_actual[df_actual["DATE"] == previous_date]

    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered_actual['USEP'],
            mode = 'lines',
            name = 'Actual',
            line = {'color': '#3E9651'}
            ))

    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered['Forecast_Baseline'],
            mode = 'lines',
            name = 'Baseline Forecast',
            line = {'color': 'red',
                    'dash':'dashdot'}
            ))

    traces.append(
            go.Scatter(
                x = df_dayahead['PERIOD_TIME'].unique(),
                y = df_filtered['Model_1'],
                mode = 'lines',
                name = 'M1 OLS Regression',
                line = {'color': 'blue',
                        'dash':'dot'}
                ))

    traces.append(
            go.Scatter(
                x = df_dayahead['PERIOD_TIME'].unique(),
                y = df_filtered['Model_2'],
                mode = 'lines',
                name = 'M2 Decision Tree',
                line = {'color': 'purple',
                        'dash':'dot'}
                ))

    return {
        'data' : traces,
        'layout': go.Layout(
            title = '',
            xaxis = {'title': 'Period (hh:mm)', 'showgrid':False, 'nticks':6}, # Hide gridlines, and show 6 x ticks
            yaxis = {'title': 'USEP ($/MWh)', 'showgrid':False},
            legend=dict(x=-.1, y=1.4),
            hovermode='x')
    }


# =============================================
#    Callback for Following Day Chart
# =============================================

@app.callback(
    Output('graph-following-day', 'figure'),
    [Input('date-picker', 'date')])
def update_output(selected_date):
    traces = []          # Traces is a list to contain data points

    new_selected_date = dt.strptime(selected_date.split(' ')[0], '%Y-%m-%d')
    following_date = new_selected_date + timedelta(days=1)
    df_filtered = df_dayahead[df_dayahead["DATE"] == following_date]
    df_filtered_actual = df_actual[df_actual["DATE"] == following_date]

    traces.append(
        go.Scatter(
            x = df_dayahead['PERIOD_TIME'].unique(),
            y = df_filtered['Forecast_Baseline'],
            mode = 'lines',
            name = 'Baseline Forecast',
            line = {'color': 'red',
                    'dash':'dashdot'}
            ))

    traces.append(
            go.Scatter(
                x = df_dayahead['PERIOD_TIME'].unique(),
                y = df_filtered['Model_1'],
                mode = 'lines',
                name = 'M1 OLS Regression',
                line = {'color': 'blue',
                        'dash':'dot'}
                ))

    traces.append(
            go.Scatter(
                x = df_dayahead['PERIOD_TIME'].unique(),
                y = df_filtered['Model_2'],
                mode = 'lines',
                name = 'M2 Decision Tree',
                line = {'color': 'purple',
                        'dash':'dot'}
                ))

    return {
        'data' : traces,
        'layout': go.Layout(
            title = '',
            xaxis = {'title': 'Period (hh:mm)', 'showgrid':False, 'nticks':6}, # Hide gridlines, and show 6 x ticks
            yaxis = {'title': 'USEP ($/MWh)', 'showgrid':False},
            legend=dict(x=-.1, y=1.4),
            hovermode='x') # x = Compare data on hover. Other option is Closest
    }






# =====================
#  Add server clause
# =====================
if __name__ == '__main__':
    app.run_server(debug=True)
