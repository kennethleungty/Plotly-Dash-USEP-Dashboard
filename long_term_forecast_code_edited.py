#==================================
#   Long Term Forecast Model
#==================================
import pandas as pd
import numpy as np
import pyforest
from scipy.stats import pearsonr
from statsmodels.api import OLS
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics
import statsmodels.api as sm
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from pylab import rcParams
import itertools
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/model_data/dem_usep.csv')
mc = pd.read_csv('data/model_data/SRMC.csv')
brent = pd.read_csv('data/model_data/DCOILBRENTEU_2.csv')
ex = pd.read_csv('data/model_data/USDSGD.csv')
hol = pd.read_csv('data/model_data/holidays_filter.csv')
ind = pd.read_csv('data/model_data/ind_prod.csv')
df['month_year'] = pd.to_datetime(df['TRADING_DATE']).dt.to_period('M')
brent['month_year'] = pd.to_datetime(brent['date']).dt.to_period('M')
mc['month_year'] = pd.to_datetime(mc['TRADING_DATE']).dt.to_period('M')
ex['month_year'] = pd.to_datetime(ex['date']).dt.to_period('M')
ind['month_year'] = pd.to_datetime(ind['month']).dt.to_period('M')
mc = mc.drop(columns=['TRADING_DATE'])
brent = brent.drop(columns=['date'])
df_monthly = df.groupby(['month_year']).mean().drop(columns=['PERIOD'])
df_monthly['peak_calc'] = df.groupby(['month_year']).max()['DEMAND']
df_monthly = df_monthly.reset_index()
brent['price'] = brent.groupby(['month_year'])['price'].transform(lambda x: x.fillna(method='pad'))
brent['price'] = brent.groupby(['month_year'])['price'].transform(lambda x: x.fillna(method='bfill'))
brent_monthly = brent.groupby(['month_year']).mean()
ex_monthly = ex.groupby(['month_year']).mean()
brent_monthly['brent_in_SGD'] = brent_monthly * ex_monthly
hol['month_year'] = pd.to_datetime(hol['date']).dt.to_period('M')
hol = hol[['hol_type', 'month_year']]
hol['hol_type'] = 1
hol = hol.groupby(['month_year']).sum()
df_monthly = df_monthly.merge(mc, on='month_year', how='left')
df_monthly = df_monthly.merge(brent_monthly, on='month_year', how='left')
df_monthly = df_monthly.merge(hol, on='month_year', how='left')
df_monthly = df_monthly.merge(ind, on='month_year', how='left').drop(columns='month')
df_monthly['hol_type'] = df_monthly['hol_type'].fillna(0)
df_monthly['USEP'][30] = df_monthly['USEP'][30]*0.7
df_monthly['month'] = df_monthly['month_year'].dt.month
df_monthly = df_monthly.join(pd.get_dummies(df_monthly['month'], drop_first=True),lsuffix='month')#.drop(columns='month')
df_monthly['reserve_r'] = (df_monthly['capacity_ccgt'] - df_monthly['peak_calc']) / df_monthly['peak_calc']
df_monthly['reserve_abs'] = df_monthly['reserve_r'] * df_monthly['DEMAND']
sc = StandardScaler()
numerical = ['DEMAND','peak_calc', 'ccgt_mc', 'steam_mc', 'capacity', 'capacity_ccgt',
             'capacity_steam', 'price', 'brent_in_SGD', 'reserve_abs']
df_monthly_sc = df_monthly.copy()
df_monthly_sc[numerical] = sc.fit_transform(df_monthly_sc[numerical])
df_monthly_sc['brent_lag'] = df_monthly_sc['brent_in_SGD'].shift(periods=2)
brent_2012 = pd.read_csv('data/model_data/DCOILBRENTEU_2012.csv')
brent_2012['month_year'] = pd.to_datetime(brent_2012['date']).dt.to_period('M')
brent_2012 = brent_2012.drop(columns=['date'])
brent_2012['price'] = brent_2012.groupby(['month_year'])['price'].transform(lambda x: x.fillna(method='pad'))
brent_2012['price'] = brent_2012.groupby(['month_year'])['price'].transform(lambda x: x.fillna(method='bfill'))
brent_2012_monthly = brent_2012.groupby(['month_year']).mean()
df_monthly['brent_lag'] = df_monthly['brent_in_SGD'].shift(periods=2)
df_monthly['brent_lag'][0] = brent_2012_monthly['price'][0]
df_monthly['brent_lag'][1] = brent_2012_monthly['price'][1]
sc = StandardScaler()
numerical = ['DEMAND','peak_calc', 'ccgt_mc', 'steam_mc', 'capacity', 'capacity_ccgt', 'capacity_steam',
             'price', 'brent_in_SGD', 'reserve_r', 'reserve_abs', 'brent_lag']
df_monthly_sc = df_monthly.copy()
df_monthly_sc[numerical] = sc.fit_transform(df_monthly_sc[numerical])
df_monthly_sc = df_monthly_sc.set_index('month_year')
df_monthly_train = df_monthly[:-6] #2013-2018 data inclusive
df_monthly_test = df_monthly[-6:] #2019 data
y_train_m = df_monthly_train.USEP
X_train_m = df_monthly_train[['brent_lag', 'capacity']]
X_test_m = df_monthly_test[['brent_lag', 'capacity']]
y_test_m = df_monthly_test.USEP
clf = OLS(y_train_m, X_train_m).fit()
y_train = df_monthly_train['USEP']
X_train = df_monthly_train.drop(columns=['USEP', 'month_year'])
cols = list(X_train.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X_train[cols]
    X_1 = sm.add_constant(X_1)
    model = OLS(y_train,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
clf1 = sm.OLS(y_train, df_monthly_train[cols]).fit()
rb = RobustScaler()
y_train = df_monthly_train['USEP']
X_train = df_monthly_train.drop(columns=['USEP', 'month_year'])
X_train_rb = rb.fit_transform(X_train)
X_train_sc = sc.fit_transform(X_train)
y_test = df_monthly_test['USEP']
X_test = df_monthly_test.drop(columns=['USEP', 'month_year'])
X_test_rb = rb.transform(X_test)
X_test_sc = sc.transform(X_test)
svr = GridSearchCV(SVR(gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5),
                              "kernel": ['linear', 'rbf', 'sigmoid']},scoring='neg_mean_absolute_error')
svr.fit(X_train_rb, y_train_m)
estimator = SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.01,
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
selector = RFE(estimator, 5, step=1).fit(X_train_rb, y_train_m)
pd.DataFrame(data=selector.ranking_, index = X_train.columns)
kr = GridSearchCV(KernelRidge(),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5),
                              "kernel": [ExpSineSquared(l, p)
                                         for l in np.logspace(-2, 2, 10)
                                         for p in np.logspace(0, 2, 10)]})
kr.fit(X_train_rb, y_train_m)
y = df_monthly.set_index('month_year')['peak_calc']
y.index=y.index.to_timestamp()
#rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
pred_peak = results.get_forecast(steps=48)
pred_ci = pred_peak.conf_int()
y = df_monthly.set_index('month_year')['DEMAND']
y.index=y.index.to_timestamp()
#rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
pred_dem = results.get_forecast(steps=48)
pred_ci = pred_dem.conf_int()
y = df_monthly.set_index('month_year')['value']
y.index=y.index.to_timestamp()
#rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
pred_prod = results.get_forecast(steps=48)
pred_ci = pred_prod.conf_int()
cols = ['price','brent_lag', 'brent_in_SGD', 'capacity', 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        'peak_calc', 'reserve_r', 'reserve_abs', 'DEMAND', 'value']
y_train = df_monthly_train['USEP']
X_train = df_monthly_train[cols]
X_train_rb = rb.fit_transform(X_train)
X_train_sc = sc.fit_transform(X_train)
y_test = df_monthly_test['USEP']
X_test = df_monthly_test[cols]
X_test_rb = rb.transform(X_test)
X_test_sc = sc.transform(X_test)
#rcParams['figure.figsize'] = 8, 5
kr = GridSearchCV(KernelRidge(),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5),
                              "kernel": [ExpSineSquared(l, p)
                                         for l in np.logspace(-2, 2, 10)
                                         for p in np.logspace(0, 2, 10)]})
kr.fit(X_train_rb, y_train_m)
output = pd.DataFrame(columns=['MMYYYY', 'USEP', 'TYPE'])
output['MMYYYY'] = df_monthly_sc.index
output['TYPE'][:-6] = 'Actual'
output['TYPE'][-6:] = 'Forecasted'
output['USEP'][:-6] = y_train_m
output['USEP'][-6:] = kr.best_estimator_.predict(X_test_rb)
output.USEP = output.USEP.apply(lambda x: round(x,2))
brentfuture = pd.read_csv('data/model_data/brent_future.csv')
exfuture = pd.read_csv('data/model_data/usdsgd_future.csv')
capacityfuture = pd.read_csv('data/model_data/capacity_future.csv')
brentfuture['month_year'] = pd.to_datetime(brentfuture['month']).dt.to_period('M').drop(columns='month')
exfuture['month_year'] = pd.to_datetime(exfuture['month']).dt.to_period('M').drop(columns='month')
capacityfuture['month_year'] = pd.to_datetime(capacityfuture['month']).dt.to_period('M').drop(columns='month')
brentfuture['brent_in_SGD'] = brentfuture['price']*exfuture['price']
future = brentfuture.merge(capacityfuture, on='month_year', how='left')
future['brent_lag'] = future['brent_in_SGD'].shift(periods=2)
future = future.dropna()
future = future.set_index('month_year').drop(columns=['month_x', 'month_y'])
future['DEMAND'] = pred_dem.predicted_mean.reset_index().drop(columns='index').to_numpy()
future['peak_calc'] = pred_peak.predicted_mean.reset_index().drop(columns='index').to_numpy()
future['reserve_r'] = (future['capacity'] - future['peak_calc']) / future['peak_calc']
future['reserve_abs'] = future['reserve_r']*future['DEMAND']
future['value'] = pred_prod.predicted_mean.reset_index().drop(columns='index').to_numpy()
future['month'] = future.reset_index()['month_year'].dt.month.to_numpy()
future = future.join(pd.get_dummies(future['month'], drop_first=True),lsuffix='month').drop(columns='month')
output_future = pd.DataFrame(columns=['MMYYYY', 'USEP', 'TYPE'])
output_future['MMYYYY'] = future.index
output_future['TYPE'] = 'Forecasted'
output_future['USEP'] = kr.best_estimator_.predict(rb.transform(future[cols]))
output_future.USEP = output_future.USEP.apply(lambda x: round(x,2))
output = output.append(output_future)
df_monthly = output.copy()
#output.to_csv('monthly_forecast_future_new.csv', index=False)
