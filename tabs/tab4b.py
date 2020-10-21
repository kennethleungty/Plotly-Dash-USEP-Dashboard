# =====================================
#    Tab 4a - Historical Monthly
# =====================================
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from datetime import datetime as dt
import pandas as pd

from app import df_historical, df_hist_monthly

layout = html.Div(style = {'text-align':'center'},


        children=[

                # ========================================
                #      Dropdowns for Dataset and Year
                # ========================================

                html.Div(className = 'tab4b-dataset-select',
                children = [
                dcc.Dropdown(id = 'tab4b-dataset-select',
                    options=[{'label': 'Demand & USEP', 'value': 'Demand & USEP'},
                    {'label': 'Primary Reserve', 'value': 'Primary Reserve'},
                    {'label': 'Contingency Reserve', 'value': 'Contingency Reserve'},
                    {'label': 'Regulation Reserve', 'value': 'Regulation Reserve'}],
                    value='Demand & USEP', searchable=False, clearable = False,
                    placeholder='Select Dataset',
                    style = {'width':'220px',
                            'height':'33px',
                            'verticalAlign':'bottom'}
                    )
                ]),


                html.Div(className = 'year-input-monthly',
                children = [
                dcc.Dropdown(id = 'year-input-monthly',
                    options=[{'label': i, 'value': i} for i in range(2015,2020)],
                    value=2019,    # Value needs to be number, not string like '2019'
                    searchable=False, clearable = False,
                    placeholder='Year',
                    style = {'width':'220px',
                            'height':'33px',
                            'verticalAlign':'bottom'}
                    )
                ]),

                # html.Div(
                # html.Button('Submit', id='tab4b-submit-button',
                #                 style = {'font-size':'16px',
                #                         'color':'white',
                #                         'background-color':'#F05E23',
                #                         'width':'340px'},
                #                 className = 'tab4b-submit-button')
                # ),



                # ===============================
                #         Data Table
                # ===============================

                html.Div(className = 'monthly-data-table',
                children = [
                    dash_table.DataTable(
                    id='monthly-data-table',
                    columns=[{"name": i, "id": i} for i in df_hist_monthly.columns],
                    data= df_hist_monthly.to_dict('records'),
                    hidden_columns=['MONTH','YEAR','DATASET'],
                    style_header={'backgroundColor': '#D5F5E3',
                            'fontWeight': 'bold',
                            'textAlign': 'Center'},
                    style_data = {                      # Allowing text wrapping (Overflow into multiple lines)
                                'whiteSpace': 'normal',
                                'height': 'auto'},
                    style_cell={'padding': '5px',
                            'textAlign': 'Center',
                            'font-family':'sans-serif',
                            'font-size':'13px'},
                    style_data_conditional=[{             # Alternating row background colours
                        'if': {'row_index': 'odd'},
                                'backgroundColor': '#F2F3F4'
                                }],
                    style_as_list_view=True,
                    css= [{"selector": ".show-hide", "rule": "display: none"}] # Hide toggle button
                )
                ]),

                # ===============================
                #  Line Graph (from Data Table)
                # ===============================

                html.Div([
                    dcc.Graph(id='graph-historical-monthly', className = 'graph-historical-monthly')
                ]),



    ]
)
