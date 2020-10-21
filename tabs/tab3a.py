# =====================================
#   Tab 3a - Latest Energy Market News
# =====================================
import dash_core_components as dcc
import dash_html_components as html
import dash_table

layout = html.Div(style = {'text-align':'center'},

        children=[

    # =========================================
    #   News Feed - RSS (Energy Sector News)
    # =========================================
    # Generated from https://rss.app/
    # Using Premium trial
    html.Div(
    html.P(
        html.Iframe(className = 'newsfeed',
        srcDoc='''
        <rssapp-list id="jIJXQOb8pgQdU3P3"></rssapp-list><script src="https://widget.rss.app/v1/list.js" type="text/javascript" async></script>
        '''
        )
    )
    ),

        ]
        )
