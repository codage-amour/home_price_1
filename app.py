import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("Bengaluru_House_Data.csv")
df1 = df.drop(["area_type", "availability", "society", "balcony"], axis='columns')
median_bath = math.floor(df.bath.median())
df1['bath'] = df1.bath.fillna(median_bath)
df1['size'] = df1['size'].fillna('0')
df1['rooms'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))
df1 = df1.drop(['size'], axis='columns')
median_rooms = math.floor(df1.rooms.median())
df1['rooms'] = df1.rooms.replace(0, median_rooms)
df1 = df1.dropna()
def range_to_avg(x):
    try:
        token = x.split('-')
        if len(token) == 2:
            y = float(token[0]) + (float(token[1]) - float(token[0])) / 2
            return y
        return float(x)
    except:
        return np.nan
df1['total_sqft'] = df1['total_sqft'].apply(range_to_avg)
df1['total_sqft'] = pd.to_numeric(df1['total_sqft'], errors='coerce')
df1 = df1.dropna(subset=['total_sqft'])
df1["price_per_sqft"] = df1["price"] * 100000 / df1['total_sqft']
df1['location'] = df1['location'].apply(lambda x: str(x).strip())
location_group = df1.groupby('location')['location'].agg('count')
less_than_10 = location_group[location_group <= 10]
df1['location'] = df1['location'].apply(lambda x: 'other' if x in less_than_10 else x)
#assumed that room's size can't be less than 300sqft
df1 = df1[~(df1.total_sqft / df1.rooms < 300)]
dummies = pd.get_dummies(df1.location)
dummies = dummies.astype(int)
df2 = pd.concat([df1, dummies], axis='columns')
df2 = df2.drop(['location', 'other'], axis='columns')
X = df2.drop(['price', 'price_per_sqft'], axis='columns')
y = df2['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)
clf = LinearRegression()
clf.fit(X_train, y_train)
# print(clf.score(X_test,y_test))   #low
#will try other models
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_score
# cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=150)
# print(cross_val_score(LinearRegression(),X,y,cv=cv)) #almost same
def predict_price(loc,bath,bhk,sqft):
    loc_index=np.where(X.columns==loc)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    return clf.predict([x])[0]
import pickle
with open("home_price","wb") as f:
    pickle.dump(clf,f)
#we need columns in a order as they are now
import json
columns={'data_columns':[col.lower() for col in X.columns]}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))



