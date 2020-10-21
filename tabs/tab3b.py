# =====================================
#   Tab 3b - Indices & Commodities
# =====================================

import dash_core_components as dcc
import dash_html_components as html
import dash_table

layout = html.Div(style = {'text-align':'center'},

        children=[

            # =======================================
            #        Trading View - Graph
            # =======================================

            # Use of Iframe to embed entire chunk of HTML code. Place the HTML code within the ''' of the srcDoc parameter
                html.Iframe(className = 'tradingview_graph',
                srcDoc='''
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Market Data</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
          {
          "colorTheme": "light",
          "dateRange": "3m",
          "showChart": true,
          "width": 550,
          "height": 680,
          "locale": "en",
          "largeChartUrl": "",
          "isTransparent": false,
          "plotLineColorGrowing": "rgba(33, 150, 243, 1)",
          "plotLineColorFalling": "rgba(33, 150, 243, 1)",
          "gridLineColor": "rgba(240, 243, 250, 1)",
          "scaleFontColor": "rgba(120, 123, 134, 1)",
          "belowLineFillColorGrowing": "rgba(33, 150, 243, 0.12)",
          "belowLineFillColorFalling": "rgba(33, 150, 243, 0.12)",
          "symbolActiveColor": "rgba(33, 150, 243, 0.12)",
          "tabs": [
            {
              "title": "Indices",
              "symbols": [
                {
                  "s": "OANDA:SPX500USD",
                  "d": "S&P 500"
                },
                {
                  "s": "OANDA:NAS100USD",
                  "d": "Nasdaq 100"
                },
                {
                  "s": "FOREXCOM:DJI",
                  "d": "Dow 30"
                },
                {
                  "s": "INDEX:NKY",
                  "d": "Nikkei 225"
                },
                {
                  "s": "INDEX:DEU30",
                  "d": "DAX Index"
                },
                {
                  "s": "OANDA:UK100GBP",
                  "d": "FTSE 100"
                }
              ],
              "originalTitle": "Indices"
            },
            {
              "title": "Commodities",
              "symbols": [
                {
                  "s": "CME_MINI:ES1!",
                  "d": "E-Mini S&P"
                },
                {
                  "s": "COMEX:GC1!",
                  "d": "Gold"
                },
                {
                  "s": "NYMEX:CL1!",
                  "d": "Crude Oil"
                },
                {
                  "s": "NYMEX:NG1!",
                  "d": "Natural Gas"
                }
              ],
              "originalTitle": "Commodities"
            },
            {
              "title": "Bonds",
              "symbols": [
                {
                  "s": "CME:GE1!",
                  "d": "Eurodollar"
                },
                {
                  "s": "CBOT:ZB1!",
                  "d": "T-Bond"
                },
                {
                  "s": "CBOT:UB1!",
                  "d": "Ultra T-Bond"
                },
                {
                  "s": "EUREX:FGBL1!",
                  "d": "Euro Bund"
                },
                {
                  "s": "EUREX:FBTP1!",
                  "d": "Euro BTP"
                },
                {
                  "s": "EUREX:FGBM1!",
                  "d": "Euro BOBL"
                }
              ],
              "originalTitle": "Bonds"
            },
            {
              "title": "Forex",
              "symbols": [
                {
                  "s": "FX:EURUSD"
                },
                {
                  "s": "FX:GBPUSD"
                },
                {
                  "s": "FX:USDJPY"
                },
                {
                  "s": "FX:USDCHF"
                },
                {
                  "s": "FX:AUDUSD"
                },
                {
                  "s": "FX:USDCAD"
                }
              ],
              "originalTitle": "Forex"
            }
          ]
        }
          </script>
        </div>
        <!-- TradingView Widget END -->
         '''
             ),

            # =======================================
            #        Trading View - Table
            # =======================================

            html.Iframe(className = 'tradingview_table',
            srcDoc='''
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com" rel="noopener" target="_blank"><span class="blue-text">Market Data</span></a> by TradingView</div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-quotes.js" async>
          {
          "width": 550,
          "height": 680,
          "symbolsGroups": [
            {
              "name": "Indices",
              "originalName": "Indices",
              "symbols": [
                {
                  "name": "OANDA:SPX500USD",
                  "displayName": "S&P 500"
                },
                {
                  "name": "OANDA:NAS100USD",
                  "displayName": "Nasdaq 100"
                },
                {
                  "name": "FOREXCOM:DJI",
                  "displayName": "Dow 30"
                },
                {
                  "name": "INDEX:NKY",
                  "displayName": "Nikkei 225"
                },
                {
                  "name": "INDEX:DEU30",
                  "displayName": "DAX Index"
                },
                {
                  "name": "OANDA:UK100GBP",
                  "displayName": "FTSE 100"
                }
              ]
            },
            {
              "name": "Commodities",
              "originalName": "Commodities",
              "symbols": [
                {
                  "name": "CME_MINI:ES1!",
                  "displayName": "E-Mini S&P"
                },
                {
                  "name": "CME:6E1!",
                  "displayName": "Euro"
                },
                {
                  "name": "COMEX:GC1!",
                  "displayName": "Gold"
                },
                {
                  "name": "NYMEX:CL1!",
                  "displayName": "Crude Oil"
                },
                {
                  "name": "NYMEX:NG1!",
                  "displayName": "Natural Gas"
                },
                {
                  "name": "CBOT:ZC1!",
                  "displayName": "Corn"
                }
              ]
            },
            {
              "name": "Bonds",
              "originalName": "Bonds",
              "symbols": [
                {
                  "name": "CME:GE1!",
                  "displayName": "Eurodollar"
                },
                {
                  "name": "CBOT:ZB1!",
                  "displayName": "T-Bond"
                },
                {
                  "name": "CBOT:UB1!",
                  "displayName": "Ultra T-Bond"
                },
                {
                  "name": "EUREX:FGBL1!",
                  "displayName": "Euro Bund"
                },
                {
                  "name": "EUREX:FBTP1!",
                  "displayName": "Euro BTP"
                },
                {
                  "name": "EUREX:FGBM1!",
                  "displayName": "Euro BOBL"
                }
              ]
            },
            {
              "name": "Forex",
              "originalName": "Forex",
              "symbols": [
                {
                  "name": "FX:EURUSD"
                },
                {
                  "name": "FX:GBPUSD"
                },
                {
                  "name": "FX:USDJPY"
                },
                {
                  "name": "FX:USDCHF"
                },
                {
                  "name": "FX:AUDUSD"
                },
                {
                  "name": "FX:USDCAD"
                }
              ]
            }
          ],
          "colorTheme": "light",
          "isTransparent": false,
          "locale": "en"
        }
          </script>
        </div>
        <!-- TradingView Widget END -->
         '''
             ),

            # ========================================
            #            Oil Trends Plots
            # ========================================
            html.Div(style = {'textAlign':'center', 'height':'400px'},
                children = [

            html.Iframe(className = 'wti-crude',
            srcDoc='<script type="text/javascript" src="https://www.oil-price.net/TABLE2/gen.php?lang=en"> </script><noscript> To get the WTI <a href="http://www.oil-price.net/dashboard.php?lang=en#TABLE2">oil price</a>, please enable Javascript.</noscript>'
            ),

            html.Iframe(className = 'brent-crude',
            srcDoc='<script type="text/javascript" src="https://www.oil-price.net/widgets/brent_crude_price_large/gen.php?lang=en"> </script>  <noscript> To get the BRENT <a href="http://www.oil-price.net/dashboard.php?lang=en#brent_crude_price_large">oil price</a>, please enable Javascript.</noscript>'
            ),

            html.Iframe(className = 'natural-gas',
            srcDoc='<script type="text/javascript" src="https://www.oil-price.net/widgets/natural_gas_large/gen.php?lang=en"> </script><noscript> To get the <a href="http://www.oil-price.net">natural gas price</a>, please enable Javascript.</noscript>'
            ),

            html.Iframe(className = 'gold-price',
            srcDoc='<script type="text/javascript" src="https://www.gold-quote.net/TABLE2/gen.php?lang=en"> </script><noscript> To get the <a href="http://www.gold-quote.net">gold price</a>, please enable Javascript.</noscript>'
            )
            ]),

            # html.Iframe(className = 'commodities',
            # srcDoc='<script type="text/javascript" src="https://www.oil-price.net/COMMODITIES/gen.php?lang=en"> </script> <noscript> To get <a href="http://www.oil-price.net/dashboard.php?lang=en#COMMODITIES">gold, oil and commodity prices</a>, please enable Javascript.</noscript>'
            # ),

        ]
        )
