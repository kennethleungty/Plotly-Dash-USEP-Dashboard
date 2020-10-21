# ==============================================================================
#   Python program to find forecasted weather details of Singapore with NEA API
# ==============================================================================

# Import required libraries
import json, requests
from datetime import datetime as dt
from datetime import date

# =======================================
#     Get max and min temp for today
# =======================================
today_for_weather = date.today()
today_for_weather = today_for_weather.strftime("%Y-%m-%d")

complete_url = f"https://api.data.gov.sg/v1/environment/24-hour-weather-forecast?date={today_for_weather}"

# get method of requests module, that will return response object
response = requests.get(complete_url)
x = response.json()

min_temp = x['items'][0]['general']['temperature']['low']
max_temp = x['items'][0]['general']['temperature']['high']
condition = x['items'][0]['general']['forecast'].lower()
