# =====================================
#    Tab 4a - Historical Daily
# =====================================
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from app import df_historical

layout = html.Div(style = {'text-align':'center'},

        children=[
            # html.Div(className = 'tab4a-selector-title',
            #         children = [
            #         html.H3('Make your selection here:')
            #     ]),


                # ========================================
                # Dropdowns for Dataset, Month and Year
                # ========================================

                html.Div(className = 'tab4a-dataset-select',
                children = [
                dcc.Dropdown(id = 'tab4a-dataset-select',
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

                html.Div(className = 'year-input-daily',
                children = [
                dcc.Dropdown(id = 'year-input-daily',
                    options=[{'label': i, 'value': i} for i in range(2015,2020)],
                    value=2019,    # Value needs to be number, not string like '2019'
                    searchable=False, clearable = False,
                    placeholder='Year',
                    style = {'width':'220px',
                            'height':'33px',
                            'verticalAlign':'bottom'}
                    )
                ]),


                html.Div(className = 'month-input-daily',
                children = [
                dcc.Dropdown(id = 'month-input-daily',
                    options=[{'label': 'January', 'value': 1},
                    {'label': 'February', 'value': 2}, {'label': 'March', 'value': 3},
                    {'label': 'April', 'value': 4}, {'label': 'May', 'value': 5}, {'label': 'Jun', 'value': 6},
                    {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8}, {'label': 'September', 'value': 9},
                    {'label': 'October', 'value': 10}, {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}],
                    value=1, searchable=True, clearable = False,
                    placeholder='Month',
                    style = {'width':'220px',
                            'height':'33px',
                            'verticalAlign':'bottom'},
                    )
                ]),


                # html.Div(
                # html.Button('Submit', id='tab4a-submit-button',
                #                 style = {'font-size':'16px',
                #                         'color':'white',
                #                         'background-color':'#F05E23',
                #                         'width':'460px'},
                #                 className = 'tab4a-submit-button')
                # ),



                # ===============================
                #         Data Table
                # ===============================

                html.Div(className = 'daily-data-table',
                children = [
                    dash_table.DataTable(
                    id='daily-data-table',
                    columns=[{"name": i, "id": i} for i in df_historical.columns],
                    data= df_historical.to_dict('records'),
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
                    dcc.Graph(id='graph-historical-daily', className = 'graph-historical-daily')
                ]),



    ]
)
