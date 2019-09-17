# import pandas as pd
# # Read data from file 'filename.csv'
# # (in the same directory that your python process is based)
# # Control delimiters, rows, column names with read_csv (see later)
# data = pd.read_csv("GOOG.csv")
# # # Preview the first 5 lines of the loaded data
# # print(data)
#
# close_px = data['Adj Close']
# # mavg = close_px.rolling(window=100).mean()
# #
# # print(mavg)



import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

# print(mavg)
#
# ##Graph
# import matplotlib.pyplot as plt
# from matplotlib import style
#
# # Adjusting the size of matplotlib
# import matplotlib as mpl
# mpl.rc('figure', figsize=(8, 7))
# mpl.__version__
#
# # Adjusting the style of matplotlib
# style.use('ggplot')
#
# close_px.plot(label='AAPL')
# mavg.plot(label='mavg')
# plt.legend()
# #plt.show()
#
# rets = close_px / close_px.shift(1) - 1
# rets.plot(label='return')
#
#
# ### Competitor analyzis
#
# dfcomp = web.DataReader(['AAPL', 'AMZN', 'GOOG', 'FB'],'yahoo',start=start,end=end)['Adj Close']
# #print(dfcomp)
#
# retscomp = dfcomp.pct_change()
# corr = retscomp.corr()
# print(corr)
#
# plt.scatter(retscomp.AAPL, retscomp.FB)
# plt.xlabel("Returns AAPL")
# plt.ylabel("Returns GE")
#
# pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));
import pandas
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style

import math
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

df = pandas.read_csv("data/GOOGL.csv")

dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["Date"] = pandas.to_datetime(df['Date'])
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
dfreg = dfreg.set_index('Date')


#####



# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

forecast_out = 10

forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

X = sklearn.preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)#, shuffle=False)
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)
# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


confidencereg = clfreg.score(X_test, y_test) #0.6268349679841989
confidencesvr = svr_rbf.score(X_test,y_test) #0.4571830631098365
confidenceknn = clfknn.score(X_test, y_test) #0.47697790543220403

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()