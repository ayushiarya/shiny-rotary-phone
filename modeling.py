"""
EXPLORATION CREATING AND TUNING A MODEL USING TENSORFLOW
---This example predicts cost to load based on fuel price & load from PLEXOS
   runs with 2 generators, 2 fuels, and 27 combos of high/normal/low load and 
   fuel price

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import model_selection


def norm(df, column_list):
    for col in column_list:
        df[col] = (df[col] - x_train[col].mean())/x_train[col].std()
    return df



x_df = pd.read_csv('x_data.csv').drop(columns = ['Unnamed: 0']) # read data into Pandas DataFrame from .csv

# Need to extract values for Cost to Load, Fuel A Price, Fuel B Price, & Load
# drop rows where property_name is not Price, Load, or Cost To Load
x_df = x_df[x_df['property_name'].isin(['Price', 'Load', 'Cost to Load'])]
x_df.drop_duplicates(inplace=True)

a_prices = x_df[(x_df['property_name'] == 'Price') & (x_df['child_name'] == 'A') & (x_df['collection_name'] == 'Fuel')]
b_prices = x_df[(x_df['property_name'] == 'Price') & (x_df['child_name'] == 'B') & (x_df['collection_name'] == 'Fuel')]
load_data = x_df[(x_df['property_name'] == 'Load') & (x_df['collection_name'] == 'Region')]
y_data = x_df[(x_df['property_name'] == 'Cost to Load') & (x_df['collection_name'] == 'Region')]

a_prices = a_prices[['model_name', 'value']]
a_prices.rename(columns = {'value' : 'price_A'}, inplace = True)
b_prices = b_prices[['model_name', 'value']]
b_prices.rename(columns = {'value' : 'price_B'}, inplace = True)
load = load_data[['model_name', 'value']]
load.rename(columns = {'value' : 'load'}, inplace = True)
y_data = y_data[['model_name', 'value']]
y_data.rename(columns = {'value' : 'cost_to_load'}, inplace = True)

# restructure into a DataFrame
x_data = pd.merge(a_prices, b_prices, on='model_name').merge(load, on='model_name')
x_data.set_index('model_name')
y_data.set_index('model_name')

# some data checks
print("Features:\t\t\t", x_data.columns.tolist()[1:])
print("Target:\t\t\t\t", y_data.columns.tolist()[1:])
print("Number of Observations:\t\t", x_df.shape[0])
print("Number of rows missing data: \t", (x_df.shape[0] - x_df.dropna().shape[0]))

x_data.drop(columns = ['model_name'], inplace = True)
y_data.drop(columns = ['model_name'], inplace = True)
# split data into x & y, test & train sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size = 0.7)

print("\nSummary of X Training Set:\n", x_train.describe())
print("\nSummary of Y Training Set:\n", y_train.describe())
print("\n\n\n")

# normalize training set - let's see how the results come out without this first

#normed_x_train = norm(x_train, ['price_A', 'price_B', 'load'])
#normed_x_test = norm(x_test, ['price_A', 'price_B', 'load'])

# create the model 

model = keras.Sequential([ \
                          layers.Dense(1, activation = tf.nn.relu, input_shape = [len(x_train.keys())]), \
                          layers.Dense(1) \
                          ])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])

model.build()


model.summary()

result = model.predict(x_test)


