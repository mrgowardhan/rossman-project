import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
#preprocessing
# from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


# Modelling Helpers
from sklearn.model_selection import train_test_split

# Evaluation metrics
from sklearn.metrics import r2_score as r2, mean_squared_error as mse, mean_absolute_error as mae
import math
stores_data = pd.read_csv('Rossmann Stores Data.csv')
stores = pd.read_csv('store.csv')
print('Shape of stores_data:', stores_data.shape)
stores_data.head()
stores_data.info()
stores_data.describe()
stores_data['DayOfWeek'].unique()
stores_data['DayOfWeek'].value_counts()
stores_data['Open'].unique()
print('Open:\n', stores_data['Open'].value_counts(), '\n\n')
print('Promo\n', stores_data['Promo'].value_counts(), '\n\n')
print('State Holiday\n', stores_data['StateHoliday'].value_counts(), '\n\n')
print('School Holiday\n', stores_data['SchoolHoliday'].value_counts())
stores_data['StateHoliday'].unique()
stores_data['StateHoliday'] = stores_data['StateHoliday'].apply(lambda x: 0 if x == '0' else x)
stores_data['StateHoliday'].value_counts()
df1 = stores_data.groupby(["StateHoliday", 'Open'])
# Number of stores those were open and closed on state holidays:
df1['Open'].value_counts().to_frame()

# Let's check the total sale of the stores open on state holidays:
df2 = df1[['Open', 'Sales']].sum()
df2
df3 = df2[df2["Open"] != 0]
df3
# Let's find out the total sale of store which were open:
t1 = df3['Sales'].sum()

# Let us compare with sum of sales in stores data to verify whether it is correct or not.
t2 = stores_data['Sales'].sum()
print(t1,',', t2)
# Percentage sale of different types of stores:
total = df3['Sales'].sum()
sale_a = ((5890305)/total)*100
sale_b = ((1433744)/total)*100
sale_c = ((691806)/total)*100
sale_a, sale_b, sale_c

stores_data.StateHoliday.replace({'0' : 0,
                            'a' : 1,
                            'b' : 1,
                            'c' : 1}, inplace=True)
print('State Holiday\n', stores_data['StateHoliday'].value_counts())

stores_data['SchoolHoliday'].unique()

# dealing with the date-time in data:

stores_data["Date"]=pd.to_datetime(stores_data["Date"])
stores_data["Year"]=stores_data["Date"].dt.year
stores_data["Month"]=stores_data["Date"].dt.month
stores_data["Day"]=stores_data["Date"].dt.day
stores_data["Week"] = stores_data["Date"].dt.isocalendar().week % 4

stores_data["Date"] = pd.to_datetime(stores_data["Date"])

stores_data['Quarter'] = stores_data['Date'].dt.quarter

stores_data.head(10)
print('Shape of stores:', stores.shape)
stores.head()
stores.info()
stores.describe()
stores['CompetitionOpenSinceYear'].unique()
stores['CompetitionOpenSinceYear'].value_counts()
# Let's replace 1900 & 1961 in 'CompetitionOpenSinceYear' column by mode value:
stores.CompetitionOpenSinceYear.replace(1900, int(stores.CompetitionOpenSinceYear.mode()[0]), inplace=True)
stores.CompetitionOpenSinceYear.replace(1961, int(stores.CompetitionOpenSinceYear.mode()[0]), inplace=True)

int(stores.CompetitionOpenSinceYear.mode()[0])
# Null Values in 'Competition Distance' column:
stores['CompetitionDistance'].isnull().sum()
stores[stores['CompetitionDistance'].isnull()]
stores[stores['CompetitionDistance'].isnull()]
# Let's fill the null values in 'CompetitionDistance' column by mean competition distance:
stores['CompetitionDistance'].fillna(stores['CompetitionDistance'].mean(), inplace = True)
stores.iloc[290]
stores.CompetitionOpenSinceMonth.value_counts()
# Let's fill remaining NA values with zero:
stores.CompetitionOpenSinceMonth.fillna(0, inplace=True)
stores.CompetitionOpenSinceYear.fillna(0, inplace=True)
stores.Promo2SinceWeek.fillna(0, inplace=True)
stores.Promo2SinceYear.fillna(0, inplace=True)
stores.PromoInterval.fillna(0, inplace=True)
stores.info()
# distribution of Sales has a very long tail

plt.title("Distribution of sales")
sns.distplot(stores_data['Sales'])
plt.show()

sns.barplot(x = stores['StoreType'], y = stores_data['Sales'])
plt.show()
sns.barplot(x = stores['Assortment'], y = stores_data['Sales'])
plt.show()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,4))

axis1.title.set_text('Promo vs.Sales')
axis2.title.set_text('Promo vs.Customers')
sns.barplot(x='Promo', y='Sales', data=stores_data, ax=axis1)
sns.barplot(x='Promo', y='Customers', data=stores_data, ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StateHoliday', y='Sales', data= stores_data, ax=axis1)
sns.barplot(x='StateHoliday', y='Customers', data= stores_data, ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='SchoolHoliday', y='Sales', data= stores_data, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Customers', data= stores_data, ax=axis2)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.countplot(x='Open',hue='DayOfWeek', data=stores_data,palette="husl", ax=axis1)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='DayOfWeek', y='Sales', data=stores_data, order=[1,2,3,4,5,6,7], ax=axis1)
sns.barplot(x='DayOfWeek', y='Customers', data=stores_data, order=[1,2,3,4,5,6,7], ax=axis2)

axis = stores_data.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average sales by day of the week')
axis = stores_data.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average number of customers per day of the week')

stores_data.boxplot(column='Sales', by='Year',)
plt.show()

stores_data.boxplot(column='Sales', by='Month',)
plt.show()

axis = stores_data.groupby('Month')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average sales per month')

axis = stores_data.groupby('Month')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average customers per month')

axis = stores_data.groupby('Day')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average sales per day')

axis = stores_data.groupby('Day')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average number of customers per day')

print(stores_data.columns)
print('\n')
print(stores.columns)

df1 = pd.merge(stores_data, stores, how='left', on='Store')
df1.shape
df1.head(2)
df1.info()
df1.isna().sum()
df1['PromoInterval'].value_counts()
df1.loc[df1["PromoInterval"] == 0, "PromoInterval"] = "No Promo"
df1['PromoInterval'].value_counts()
rdf = pd.get_dummies(df1, columns=['Assortment', 'StoreType', 'PromoInterval'],
                               prefix=['Assortment', 'StoreType', 'PromoInterval']
)
rdf.shape
rdf.head()
rdf.info()
# Dropping unwanted columns:

rdf.drop(['Store','Date'], axis = 1, inplace = True)
# Creating dependent/target and independent variables:

target_col = 'Sales'
input_cols = rdf.columns.drop(target_col)
input_cols
# Spliting data into training and testing dataset:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(rdf[input_cols],
                                                   rdf[target_col],
                                                   test_size=0.2,
                                                   random_state=1)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
x_train[0:10]
# Data Scaling using Standard Scaler:
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
scaled_x_train[0:10]
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(scaled_x_train, y_train)
# Predicting test values:
y_pred = regressor.predict(scaled_x_test)
y_pred
# After building the model we are comparing the actual and the predicted values in this code:

data = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
data
# Performance of the model
r2s_1 = r2(y_test,y_pred)
mae1 = mae(y_test,y_pred)
rmse1 = math.sqrt(mse(y_test,y_pred))
print('Performance of Linear Regression Model:')
print('-'*40)
print('r2_score:',r2s_1)
print('Mean absolute error: %.2f' % mae1)
print('Root mean squared error: ', rmse1)

# Building XGBoost Regressor Model:
xgb = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)
xgb.fit(scaled_x_train,y_train)

y_predict = xgb.predict(scaled_x_test)

# Performance of the model
r2s_2 = r2(y_test,y_predict)
mae2 = mae(y_test,y_predict)
rmse2 = math.sqrt(mse(y_test,y_predict))
print('Performance of XGBoost Regressor Model:')
print('-'*40)
print('r2_score:',r2s_2)
print('Mean absolute error: %.2f' % mae2)
print('Root mean squared error: ', rmse2)

# Building Decesion Tree Regressor Model:

model = DecisionTreeRegressor()
model.fit(scaled_x_train,y_train)

y_predict = model.predict(scaled_x_test)

# Performance of the model
r2s_3 = r2(y_test,y_predict)
mae3 = mae(y_test,y_predict)
rmse3 = math.sqrt(mse(y_test,y_predict))
print('Performance of Decesion Tree Model:')
print('-'*40)
print('r2_score:',r2s_3)
print('Mean absolute error: %.2f' % mae3)
print('Root mean squared error: ', rmse3)

from sklearn.ensemble import RandomForestRegressor
# Building Random Forest Regressor Model:

random_forest_model = RandomForestRegressor(n_estimators=100)
random_forest_model.fit(scaled_x_train,y_train)

y_predict = random_forest_model.predict(scaled_x_test)

# Performance of the model
r2s_4 = r2(y_test,y_predict)
mae4 = mae(y_test,y_predict)
rmse4 = math.sqrt(mse(y_test,y_predict))
print('Performance of Random Forest Regression Model:')
print('-'*40)
print('r2_score:', r2s_4)
print('Mean absolute error: %.2f' % mae4)
print('Root mean squared error: ', rmse4)

result = {'Model': ['Linear Regression', 'XGBoost', 'Decesion Tree', 'Random Forest'],
          'R2_Score': [r2s_1, r2s_2, r2s_3, r2s_4],
          'MAE': [mae1, mae2, mae3, mae4],
          'RMSE': [rmse1, rmse2, rmse3, rmse4]}

result_df = pd.DataFrame(result)
result_df

#Lets Find Importance of each Feature
feature_importance = random_forest_model.feature_importances_
# Lets make a dataframe consists of features and values
columns = list(x_train.columns)
feature_importance_df = pd.DataFrame({'Features':columns, 'Importance':feature_importance})
feature_importance_df.set_index('Features', inplace=True)
feature_importance_df
feature_importance_df.sort_values(by=["Importance"], inplace=True, ascending=False)
feature_importance_df

# Feature Importance
Features = feature_importance_df.index

plt.figure(figsize=(15,6))
sns.barplot(y= Features, x=feature_importance_df['Importance'], data = feature_importance_df ).set(title='Feature Importance')
plt.xticks(rotation=90)
plt.show()
