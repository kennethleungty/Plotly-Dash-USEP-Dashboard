# =====================================
#   Tab 3c -   Google Trends
# =====================================

import dash_core_components as dcc
import dash_html_components as html
import dash_table

layout = html.Div(style = {'text-align':'center'},

        children=[

    # ========================================
    #           Google Trends Plot
    # ========================================
    html.Div([
    html.Iframe(className = 'google-trends-oil-sg',
    srcDoc='<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2051_RC11/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"oil","geo":"SG","time":"today 1-m"}],"category":0,"property":""}, {"exploreQuery":"date=today%201-m&geo=SG&q=oil","guestPath":"https://trends.google.com:443/trends/embed/"}); </script> '
    ),

    html.Iframe(className = 'google-trends-gas-sg',
    srcDoc='<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2051_RC11/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"gas","geo":"SG","time":"today 1-m"}],"category":0,"property":""}, {"exploreQuery":"date=today%201-m&geo=SG&q=oil","guestPath":"https://trends.google.com:443/trends/embed/"}); </script> '
    ),

    html.Iframe(className = 'google-trends-climatechange-sg',
    srcDoc='<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2051_RC11/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"climate change","geo":"SG","time":"today 1-m"}],"category":0,"property":""}, {"exploreQuery":"date=today%201-m&geo=SG&q=oil","guestPath":"https://trends.google.com:443/trends/embed/"}); </script> '
    ),

    html.Iframe(className = 'google-trends-comparator-sg',
    srcDoc='<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2051_RC11/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"/m/0bp_wy","geo":"SG","time":"today 12-m"},{"keyword":"Gas","geo":"SG","time":"today 12-m"},{"keyword":"/m/0d063v","geo":"SG","time":"today 12-m"},{"keyword":"Hot Weather","geo":"SG","time":"today 12-m"}],"category":0,"property":""}, {"exploreQuery":"geo=SG&q=%2Fm%2F0bp_wy,Gas,%2Fm%2F0d063v,Hot%20Weather&date=today 12-m,today 12-m,today 12-m,today 12-m","guestPath":"https://trends.google.com:443/trends/embed/"}); </script>'
    ),
    ]),


        ]
        )
