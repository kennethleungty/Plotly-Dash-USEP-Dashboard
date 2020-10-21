# =====================================
#    Tab 4a - Historical Yearly
# =====================================
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from datetime import datetime as dt
import pandas as pd

from app import df_historical, df_hist_monthly, df_hist_annual


layout = html.Div(style = {'text-align':'center'},

        children=[

                # ========================================
                #      Dropdowns for Dataset and Year
                # ========================================

                html.Div(className = 'tab4c-dataset-select',
                children = [
                dcc.Dropdown(id = 'tab4c-dataset-select',
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

                # html.Div(
                # html.Button('Submit', id='tab4c-submit-button',
                #                 style = {'font-size':'16px',
                #                         'color':'white',
                #                         'background-color':'#F05E23',
                #                         'width':'220px'},
                #                 className = 'tab4c-submit-button')
                # ),




                # ===============================
                #         Data Table
                # ===============================

                html.Div(className = 'annual-data-table',
                children = [
                    dash_table.DataTable(
                    id='annual-data-table',
                    columns=[{"name": i, "id": i} for i in df_hist_annual.columns],
                    data= df_hist_annual.to_dict('records'),
                    hidden_columns=['MONTH','DATASET'],
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
                    dcc.Graph(id='graph-historical-yearly', className = 'graph-historical-yearly')
                ]),



    ]
)
