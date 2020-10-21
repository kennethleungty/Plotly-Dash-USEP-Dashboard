# =====================================
#          Tab 3 - Trends
# =====================================
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
from app import app
from tabs import tab3a, tab3b, tab3c, tab3d, map
from dash.dependencies import Input, Output, State

# ==========================
#    Creating Tab Layout
# ==========================

layout = html.Div(  # This text align helps to center the following graphs

    children=[

    # =====================
    #   Tab Header Title
    # =====================

    html.H2('Market Indicators',
     style = {'color':'white',
             'textAlign':'Center',
             'background-color': '#293796',
             'height':'28px',
             'padding':'6px'}),


    # =====================
    #         Tabs
    # =====================
    html.P(            # This html.P is to create white spacing from the blocks above
        html.P(dcc.Tabs(id="tab3-tabs", value='map', children=[
             # dcc.Tab(label='Upcoming Advisories', value='map',
             #    className = 'map'), # Hide from view
             # dcc.Tab(label='Latest News', value='tab3a',
             #    className = 'tab3a'),  # Hide from view
             dcc.Tab(label='Indices & Commodities', value='tab3b',
                className = 'tab3b'),
             dcc.Tab(label='Google Trends', value='tab3c',
                className = 'tab3c'),
            dcc.Tab(label='Model Feature Trends', value='tab3d',
                className = 'tab3d'),
         ],colors={
            "border": "white",
            "primary": "red",
            "background": "#F8F9F9"
        }))
            ),
         html.Div(id='tab3-content'),

])


# =======================================
#       Callback for Tab 3 tabs
# =======================================

@app.callback(Output('tab3-content', 'children'),
    [Input('tab3-tabs', 'value')])
def render_content(tab):
    if tab == 'map':
        return map.layout

    elif tab == 'tab3a':
        return tab3a.layout

    elif tab == 'tab3b':
        return tab3b.layout

    elif tab == 'tab3c':
        return tab3c.layout

    elif tab == 'tab3d':
        return tab3d.layout
