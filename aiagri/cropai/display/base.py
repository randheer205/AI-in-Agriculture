import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
df_tr = pd.read_csv('train.csv')
df_tr.drop(df_tr.index[1461],inplace=True)
df_tr.tail()

df_te = pd.read_csv('test.csv')
df_te.head()

df = pd.concat([df_tr,df_te],ignore_index = True)
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
month = [int(d.split('-')[1]) for d in df['date']]
df['month'] = np.array(month)
df['month'] = df['month'].astype('str')
df_ = pd.get_dummies(df['month'],drop_first = True)
df.drop('month',inplace = True,axis = 1)
df = pd.concat([df,df_],axis=1)


winter = [12,1,2]
summer = [3,4,5]
monsoon = [6,7,8,9]
autumn = [10,11]

season = []
for i in month:
  if i in winter:
    season.append('winter')
  elif i in summer:
    season.append('summer')
  elif i in monsoon:
    season.append('monsoon')
  else:
    season.append('autumn')

df['season'] = np.array(season)
df_ = pd.get_dummies(df['season'],drop_first = True)
df.drop('season',inplace = True,axis = 1)

df = pd.concat([df,df_],axis=1)
df.tail()
df['date'] = pd.to_datetime(df['date'])
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat["meantemp"])>0.0]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
continuous_features = [feature for feature in df.columns if len(df[feature].unique())>10 and df[feature].dtype != 'object' and 'date' not in feature]
continuous_features,len(continuous_features)
sk = df[continuous_features].apply(lambda x:skew(x)).sort_values(ascending = False)
sk = pd.DataFrame(sk)
def ret():
    print(sk[0])
    return sk[0]
ch = [0,0.03,0.05,0.08,0.1,0.13,0.15]
df__ = pd.DataFrame()
for choice in ch:
    df_ = pd.DataFrame(skew(boxcox1p(df[continuous_features],choice)),columns=[choice],index = continuous_features)
    df__ = pd.concat([df__,df_],axis = 1)
    
df__ = pd.concat([pd.DataFrame(skew(df[continuous_features]),columns = ['Org'],index = continuous_features),df__],axis = 1)


skew_result = {}
for i in df__.index:
    min_ = 'Org'
    for j in df__.columns:
        if df__.loc[i,j]>=0 and df__.loc[i,j]<df__.loc[i,min_]:
            min_ = j
            
    skew_result[i] = min_
    

print(skew_result)
skew_result = {k:v for k,v in skew_result.items() if v != 'Org'}
df_ = df.copy()
df_train = df_.iloc[:-20,:]
df_test = df_.iloc[-20:,:]
df_train.drop('date',axis = 1,inplace = True)
df_train['label'] = df_train['meantemp'].shift(-20)
df_train_test = df_train.iloc[-20:,:]
df_train = df_train.iloc[:-20,:]
x_train = df_train.drop('label',axis = 1)
y_train = df_train['label']
x_train_test = df_train_test.drop('label',axis = 1)
def rsc(x_train,x_train_test):
  sc = RobustScaler()
  x_train[['humidity','wind_speed','meanpressure','monsoon','summer','winter']] = sc.fit_transform(x_train[['humidity','wind_speed','meanpressure','monsoon','summer','winter']])
  x_train_test[['humidity','wind_speed','meanpressure','monsoon','summer','winter']] = sc.transform(x_train_test[['humidity','wind_speed','meanpressure','monsoon','summer','winter']])
  return x_train,x_train_test
  x_train,x_train_test = rsc(x_train,x_train_test)
model = XGBRegressor(n_estimators=300,learning_rate=0.05)
model.fit(x_train,y_train)
pre = model.predict(x_train_test)
plt.plot_date(df_test['date'],df_test['meantemp'],'-')
plt.plot_date(df_test['date'],pre,'-')
plt.xticks(rotation=90)
