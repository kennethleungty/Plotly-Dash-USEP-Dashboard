# ========= Dashboard  ===========
#      App Page (app.py)
# ================================

# ======== Import libraries ============
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import dash_auth

# Libraries for machine learning model
# import pyforest
# from scipy.stats import pearsonr
# from statsmodels.api import OLS
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn import metrics
# import statsmodels.api as sm
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_selection import RFE
# from pylab import rcParams
# import itertools
# import warnings
# warnings.filterwarnings("ignore")


# ==== Authentication ===

# USERNAME_PASSWORD_PAIRS = [['MSBA=E=MC2', 'MSBA=E=MC2']]

# ======================================
#           Import Datasets
# ======================================

# ======== Day Ahead Datasets ============
# Ensure that the date column is named as DATE (and NOT TRADING_DATE)
df_dayahead = pd.read_csv('data/Short_Term_final.csv')  # df is for day ahead forecasts
df_actual = df_dayahead.copy()
df_actual = df_actual[['DATE','PERIOD','USEP']]


# Reading in the dataset for advisory notices and alerts
df_advisory = pd.read_csv('data/advisory_day_2_ascending.csv')
df_alerts = pd.read_csv('data/Alerts_v2.csv')

# ======== Long Term Forecast Datasets ============
# df_monthly = pd.read_csv('data/monthly_forecast_future.csv') # Comment this out, use output of model
df_monthly_actual = pd.read_csv('data/monthly_actual.csv')
df_monthly_modified = pd.read_csv('data/monthly_forecast_future_modified.csv')

# ================ Other Datasets ====================
df_historical = pd.read_csv('data/Historical_Data_2015_to_2019.csv')


# ======= Converting date to datetime format accordingly ==========
# Important to use dayfirst = True. See https://stackoverflow.com/questions/50367656/python-pandas-pandas-to-datetime-is-switching-day-month-when-day-is-less-t

df_dayahead["DATE"] = pd.to_datetime(df_dayahead["DATE"], dayfirst = True)
df_actual["DATE"] = pd.to_datetime(df_actual["DATE"], dayfirst = True)
df_alerts['Date'] = pd.to_datetime(df_alerts['Date'], dayfirst = True)
df_advisory['Date'] = pd.to_datetime(df_advisory['Date'], dayfirst = True)
df_historical['DATE'] = pd.to_datetime(df_historical['DATE'], dayfirst = True)

# Modifying the date column of df_historical (Strip T00:00:00)
df_historical['DATE'] = pd.DatetimeIndex(df_historical['DATE']).strftime("%Y/%m/%d")

# Rounding values to 3 decimal places
df_historical['REQUIREMENT/DEMAND (MW)'] = df_historical['REQUIREMENT/DEMAND (MW)'].round(3)
df_historical['PRICE ($/MWh)'] = df_historical['PRICE ($/MWh)'].round(2)


# === Creating MONTHLY dataset from historical dataset ===
df_hist_monthly = df_historical.copy()
df_hist_monthly['DATE'] = pd.to_datetime(df_hist_monthly['DATE'])
df_hist_monthly['DATE (YYYY/MM)'] = df_hist_monthly['DATE'].dt.strftime('%Y/%m')
df_hist_monthly.drop(columns = ['DATE','MONTH'])

# Group by MM-YYYY
df_hist_monthly = df_hist_monthly.groupby(['DATE (YYYY/MM)','DATASET'], as_index = False).mean()

# Rounding values to fewer decimal places
df_hist_monthly['REQUIREMENT/DEMAND (MW)'] = df_hist_monthly['REQUIREMENT/DEMAND (MW)'].round(3)
df_hist_monthly['PRICE ($/MWh)'] = df_hist_monthly['PRICE ($/MWh)'].round(2)


# === Creating ANNUAL dataset from historical dataset ===
df_hist_annual = df_historical.copy()
df_hist_annual.drop(columns = ['DATE','MONTH'])

# Group by YEAR
df_hist_annual = df_hist_annual.groupby(['YEAR','DATASET'], as_index = False).mean()

# Rounding values to fewer decimal places
df_hist_annual['REQUIREMENT/DEMAND (MW)'] = df_hist_annual['REQUIREMENT/DEMAND (MW)'].round(3)
df_hist_annual['PRICE ($/MWh)'] = df_hist_annual['PRICE ($/MWh)'].round(2)


# ==================================

# ======== Running the app =========
app = dash.Dash(__name__)
app.title = 'NUS EMC USEP Dashboard'
# auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)

app.config.suppress_callback_exceptions = True
server = app.server
