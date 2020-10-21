# =====================================
#    Tab 4 - Historical Average Data
# =====================================
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from app import app, server, df_historical, df_hist_monthly, df_hist_annual
from tabs import tab4a, tab4b, tab4c
from datetime import datetime as dt
import pandas as pd
from plotly.subplots import make_subplots

layout = html.Div(

    children=[

        # ========================================
        #           Header Title
        # ========================================

        html.H2('Historical Data',
            style = {'color':'white',
             'textAlign':'Center',
             'background-color': '#293796',
             'height':'28px',
             'padding':'6px'}),


        # =====================
        #         Tabs
        # =====================
        html.P(            # This html.P is to create white spacing from the blocks above
         html.P(dcc.Tabs(id="tab4-tabs", value='tab4a', children=[
             dcc.Tab(label='Daily Average', value='tab4a',
                className = 'tab4a'),
             dcc.Tab(label='Monthly Average', value='tab4b',
                className = 'tab4b'),
             dcc.Tab(label='Annual Average', value='tab4c',
                className = 'tab4c'),
         ],colors={
            "border": "white",
            "primary": "red",
            "background": "#F8F9F9"
        }))
            ),
         html.Div(id='tab4-content'),


        # =====================
        #    Download Button
        # =====================
         html.Div(style = {'textAlign': 'center'},
            children = [
                html.A([
                html.Img(
                    src='https://i.ibb.co/R3ZBm8L/download-data.png',
                    style={
                    'height' : '12%',
                    'width' : '12%',
                    # 'float' : 'right',
                    # 'position' : 'relative',
                    # 'padding-top' : 0,
                    # 'padding-right' : 0
                })
                ], href='https://www.emcsg.com/marketdata/priceinformation',
                target='_blank')  # Open link in new tab
            ]

          )
    ]
)

# =======================================
#  Callback for Historical Data tabs
# =======================================

@app.callback(Output('tab4-content', 'children'),
    [Input('tab4-tabs', 'value')])
def render_content(tab):
    if tab == 'tab4a':
        return tab4a.layout

    elif tab == 'tab4b':
        return tab4b.layout

    elif tab == 'tab4c':
        return tab4c.layout


# =======================================
#  Callback for Data Table (Daily)
# =======================================

# # Using states and submit button
# @app.callback(
#     Output('daily-data-table', 'data'),
#     [Input('tab4a-submit-button','n_clicks')],
#     [State('tab4a-dataset-select', 'value'),
#     State('month-input-daily', 'value'),
#     State('year-input-daily', 'value')])
#
# def update_daily_table(n_clicks, dataset, month, year):
#     # if n_clicks is None:
#     #     return df_historical.to_dict("rows")
#     # else:
#     df_hist_modi = df_historical.copy()
#     df_hist_modi = df_hist_modi.loc[(df_hist_modi['DATASET'] == dataset) &
#                     (df_hist_modi['MONTH'] == month) &
#                     (df_hist_modi['YEAR'] == year)]
#     return df_hist_modi.to_dict("rows")


@app.callback(
    Output('daily-data-table', 'data'),
    [Input('tab4a-dataset-select', 'value'),
    Input('month-input-daily', 'value'),
    Input('year-input-daily', 'value')])

def update_daily_table(dataset, month, year):
    # if n_clicks is None:
    #     return df_historical.to_dict("rows")
    # else:
    df_hist_modi = df_historical.copy()
    df_hist_modi = df_hist_modi.loc[(df_hist_modi['DATASET'] == dataset) &
                    (df_hist_modi['MONTH'] == month) &
                    (df_hist_modi['YEAR'] == year)]
    return df_hist_modi.to_dict("rows")

# =======================================
#     Callback for Graph (Daily)
# =======================================

@app.callback(
    Output('graph-historical-daily', 'figure'),
    [Input('tab4a-dataset-select', 'value'),
    Input('month-input-daily', 'value'),
    Input('year-input-daily', 'value')])

def update_daily_graph(dataset, month, year):
    df_hist_modi = df_historical.copy()
    df_hist_modi = df_hist_modi.loc[(df_hist_modi['DATASET'] == dataset) &
                    (df_hist_modi['MONTH'] == month) &
                    (df_hist_modi['YEAR'] == year)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace1 = go.Scatter(
                x = df_hist_modi['DATE'].unique(),
                y = df_hist_modi['PRICE ($/MWh)'],
                mode = 'markers+lines',
                name = 'Price ($/MWh)',
                line = {'color': '#D35400'},
                )

    trace2 = go.Scatter(
                x = df_hist_modi['DATE'].unique(),
                y = df_hist_modi['REQUIREMENT/DEMAND (MW)'],
                mode = 'markers+lines',
                name = 'Requirement / Demand (MW)',
                line = {'color': '#2980B9'},
                )

    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace2, secondary_y=True)

    fig.layout.update(height=500,
                        title = f'{dataset} in {year}/{month}',
                        title_font = {'size':16},
                        xaxis = {'showgrid':False, 'nticks' : 5},
                        yaxis = {'showgrid':False},
                        plot_bgcolor = 'white',
                        margin=dict(t=50, b=100, pad=3),
                        legend_orientation="h",
                        hovermode='x')

    # Set x-axis and y-axis titles
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="<b>Price</b> ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Requirement/Demand</b> (MW))", secondary_y=True)

    return fig


# =======================================
#  Callback for Data Table (Monthly)
# =======================================

@app.callback(
    Output('monthly-data-table', 'data'),
    [Input('tab4b-dataset-select', 'value'),
    Input('year-input-monthly', 'value')])

def update_monthly_table(dataset, year):
    # if n_clicks is None:
    #     return df_hist_monthly.to_dict("rows")
    # else:
    df_hist_modi_mth = df_hist_monthly.copy()
    df_hist_modi_mth = df_hist_modi_mth.loc[(df_hist_modi_mth['DATASET'] == dataset) &
                    (df_hist_modi_mth['YEAR'] == year)]
    return df_hist_modi_mth.to_dict("rows")

# =======================================
#     Callback for Graph (Monthly)
# =======================================

@app.callback(
    Output('graph-historical-monthly', 'figure'),
    [Input('tab4b-dataset-select', 'value'),
    Input('year-input-monthly', 'value')])

def update_monthly_graph(dataset, year):
    df_hist_modi_mth = df_hist_monthly.copy()
    df_hist_modi_mth = df_hist_modi_mth.loc[(df_hist_modi_mth['DATASET'] == dataset) &
                    (df_hist_modi_mth['YEAR'] == year)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace1 = go.Scatter(
                x = df_hist_modi_mth['DATE (YYYY/MM)'].unique(),
                y = df_hist_modi_mth['PRICE ($/MWh)'],
                mode = 'markers+lines',
                name = 'Price ($/MWh)',
                line = {'color': '#D35400'},
                )

    trace2 = go.Scatter(
                x = df_hist_modi_mth['DATE (YYYY/MM)'].unique(),
                y = df_hist_modi_mth['REQUIREMENT/DEMAND (MW)'],
                mode = 'markers+lines',
                name = 'Requirement / Demand (MW)',
                line = {'color': '#2980B9'},
                )

    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace2, secondary_y=True)

    fig.layout.update(height=500,
                        title = f'{dataset} in {year}',
                        title_font = {'size':16},
                        xaxis = {'showgrid':False, 'nticks' : 12},
                        yaxis = {'showgrid':False},
                        plot_bgcolor = 'white',
                        margin=dict(t=50, b=100, pad=3),
                        legend_orientation="h",
                        hovermode='x')

    # Set x-axis and y-axis titles
    fig.update_xaxes(title_text="Year/Month")
    fig.update_yaxes(title_text="<b>Price</b> ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Requirement/Demand</b> (MW))", secondary_y=True)

    return fig


# =======================================
#     Callback for Data Table (Annual)
# =======================================

@app.callback(
    Output('annual-data-table', 'data'),
    [Input('tab4c-dataset-select', 'value')])

def update_yearly_table(dataset):
    df_hist_modi_yr = df_hist_annual.copy()
    df_hist_modi_yr = df_hist_modi_yr.loc[(df_hist_modi_yr['DATASET'] == dataset)]
    return df_hist_modi_yr.to_dict("rows")


# =======================================
#     Callback for Graph (Annual)
# =======================================

@app.callback(
    Output('graph-historical-yearly', 'figure'),
    [Input('tab4c-dataset-select', 'value')])

def update_yearly_graph(dataset):
    df_hist_modi_yr = df_hist_annual.copy()
    df_hist_modi_yr = df_hist_modi_yr.loc[(df_hist_modi_yr['DATASET'] == dataset)]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace1 = go.Scatter(
                x = df_hist_modi_yr['YEAR'].unique(),
                y = df_hist_modi_yr['PRICE ($/MWh)'],
                mode = 'markers+lines',
                name = 'Price ($/MWh)',
                line = {'color': '#D35400'},
                )

    trace2 = go.Scatter(
                x = df_hist_modi_yr['YEAR'].unique(),
                y = df_hist_modi_yr['REQUIREMENT/DEMAND (MW)'],
                mode = 'markers+lines',
                name = 'Requirement / Demand (MW)',
                line = {'color': '#2980B9'},
                )

    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace2, secondary_y=True)

    fig.layout.update(height=500,
                        title = dataset,
                        title_font = {'size':16},
                        xaxis = {'showgrid':False, 'nticks' : 5},
                        yaxis = {'showgrid':False},
                        plot_bgcolor = 'white',
                        margin=dict(t=50, b=100, pad=0),
                        legend_orientation="h",
                        hovermode='x')

    # Set x-axis and y-axis titles
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="<b>Price</b> ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Requirement/Demand</b> (MW))", secondary_y=True)

    return fig
