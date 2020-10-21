# =====================================
#   Tab 3d - Local Demand & USEP Trends
# =====================================
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import base64

image_filename = 'assets/local_demand_usep_trends.png' # replace with your own image
encoded_img = base64.b64encode(open(image_filename, 'rb').read())

layout = html.Div(style = {'text-align':'center'},

        children=[

        html.Img(src='data:image/png;base64,{}'.format(encoded_img.decode()),
        style={'height':'86%', 'width':'86%'},
        className="usep-demand-trends-tab4")
        ]
        )
