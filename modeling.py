"""
EXPLORATION CREATING AND TUNING A MODEL USING TENSORFLOW
---This example predicts cost to load based on fuel price & load from PLEXOS
   runs with 2 generators, 2 fuels, and 27 combos of high/normal/low load and 
   fuel price

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Dense, Activation
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.compat.v1.estimator import LinearRegressor
from tensorflow.python.data import Dataset
import math



# READ IN AND PREP DATA
    
x_df = pd.read_csv('x_data.csv').drop(columns = ['Unnamed: 0']) # read data into Pandas DataFrame from .csv

# Need to extract values for Cost to Load, Fuel A Price, Fuel B Price, & Load
# drop rows where property_name is not Price, Load, or Cost To Load
x_df = x_df[x_df['property_name'].isin(['Price', 'Load', 'Cost to Load'])]
x_df.drop_duplicates(inplace=True)

# drop rows corresponding to Region Price data. (If we can get .json to grab correctly, won't have to do this.)
x_df = x_df.loc[~((x_df['collection_name']=='Region') & (x_df['property_name']=='Price'))]

a_prices = x_df[(x_df['property_name'] == 'Price') & (x_df['child_name'] == 'A') & (x_df['collection_name'] == 'Fuel')]
b_prices = x_df[(x_df['property_name'] == 'Price') & (x_df['child_name'] == 'B') & (x_df['collection_name'] == 'Fuel')]
load = x_df[(x_df['property_name'] == 'Load') & (x_df['collection_name'] == 'Region')]
cost_to_load = x_df[(x_df['property_name'] == 'Cost to Load') & (x_df['collection_name'] == 'Region')]


a_prices = a_prices[['model_name', 'value']]
a_prices.rename(columns = {'value' : 'price_A'}, inplace = True)

b_prices = b_prices[['model_name', 'value']]
b_prices.rename(columns = {'value' : 'price_B'}, inplace = True)

load = load[['model_name', 'value']]
load.rename(columns = {'value' : 'load'}, inplace = True)

cost_to_load = cost_to_load[['model_name', 'value']]
cost_to_load.rename(columns = {'value' : 'cost_to_load'}, inplace = True)

df = a_prices.merge(b_prices, on='model_name')
df = df.merge(load, on='model_name')
df = df.merge(cost_to_load, on='model_name')

# DataFrame, 'data', has  27 rows, 5 columns (model_name, price_A, price_B, load, cost_to_load) and an index column

x_data = df[['price_A','price_B','load']]
y_data = df[['cost_to_load']]

# some data checks
print("Features:\t\t\t", x_data.columns.tolist(), " (",len(x_data.columns.tolist()),")")
print("Target:\t\t\t\t", y_data.columns.tolist(), " (", len(y_data.columns.tolist()),")")
print("Number of Observations:\t\t", x_df.shape[0])
print("Number of rows missing data: \t", (x_df.shape[0] - x_df.dropna().shape[0]))



# MODELING
    
# Scaling the data

# Min-max scaling transforms the data into the range [0,1]
# This lowers the std deviations, so it causes outliers to have less of an effect
# new = (old -min)/(max - min)
# OR
# standard scaling transforms the data's distribution to be standard normal
# new = (old - mean)/(std dev)
# THIS PREPROCESSES DATA AFTER THE TEST/TRAIN SPLIT

# split data into x & y, test & train sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size = 0.8)
scaler_x = MinMaxScaler().fit(x_train)
scaler_y = MinMaxScaler().fit(y_train)
'''
scaler_x = StandardScaler().fit(x_train)
scaler_y = StandardScaler().fit(y_train)
'''
x_train=scaler_x.transform(x_train)
y_train = scaler_y.transform(y_train)
x_test = scaler_x.transform(x_test)


print("\nShape of X Training Set:\n", x_train.shape)
print("\nShape of Y Training Set:\n", y_train.shape)
print("\n\n\n")


# run a model 

def run_model(model='linear_regressor_full', x_test=x_test, y_test=y_test, x_train=x_train, y_train=y_train):
    '''
    Creates, trains, and runs a model.
    Inputs: model = 'neural_network', 'linear_regressor'
    '''
    print("Model selected: ", model)

    if model=='neural_network': # create, train, run neural network
        def create_model(activation1=tf.nn.relu, activation2=keras.activations.linear, \
                         activation3=keras.activations.linear, activation4 = keras.activations.linear, \
                         optimizer='adam', neurons=16):
            dropout_rate = 0.0 # or 0.2
            lr = 0.01
            model = keras.Sequential([ \
                layers.Dense(4, input_shape = np.shape(x_train[0,:]), activation = activation1), \
                layers.Dense(8, activation=activation2), \
                layers.Dense(neurons, activation=activation3), \
                layers.Dense(16, activation=activation3), \
                layers.Dropout(dropout_rate),  \
                layers.Dense(1) \
                ])
            model.compile(loss='mean_squared_error', \
                optimizer = optimizer, \
                metrics=['mean_absolute_error'])
            return model
        
        # create model
        model = KerasRegressor(build_fn=create_model, batch_size = 10, epochs = 75, verbose=0)
        
        # Use scikit-learn to grid search
        activation1 = [tf.nn.leaky_relu]
        activation2 = [keras.activations.linear] # tf.nn.relu & leaky_relu weren't as good
        activation3 = [ keras.activations.linear] # tf.nn.relu & leaky_relu weren't as good
        activation4 = [keras.activations.linear] # tf.nn.relu & leaky_relu weren't as good
        #learn_rate = [0.001,0.01, 0.1, 0.2, 0.3]
        #dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # can help prevent overfitting if it's an issue
        neurons = [8] # mae improved increasing up to 8 neurons and declined with greater than 8
        optimizer = ['RMSProp'] # 'adam' is not very good, 'SGD' was okay in some cases
        epochs = [150] # when the model parameters are well-tuned, more than 50-75 is unnecessary.
        batch_size = [1] # 5 or 10 does well
        param_grid = dict(epochs=epochs, batch_size=batch_size, activation1=activation1, \
                          activation2=activation2, activation3=activation3, \
                          activation4 = activation4, neurons=neurons, optimizer=optimizer)
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring = 'neg_mean_absolute_error')
        grid_result = grid.fit(x_train,y_train, verbose=0)
        
        # find best results
        print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        
        y_predict = grid_result.predict(x_test)
        #invert normalize
        y_predict = scaler_y.inverse_transform(np.array([y_predict]))
        #y_test = scaler_y.inverse_transform(y_test)
        #x_test = scaler_x.inverse_transform(x_test)
        for i in range(len(x_test)):
            print("\n\nPredicted=%s,\t Actual=%s,\t Diff=%s" % (y_predict[0][i], y_test.iloc[i]['cost_to_load'], np.abs(y_predict[0][i]-y_test.iloc[i]['cost_to_load'])))
            print('\n')
        
        history = model.fit(x_train, y_train, epochs=75,  batch_size=10, verbose=0, validation_data=(x_test, scaler_y.transform(y_test)))
        # batch size needs to be <10 (small) here since we don't have a large sample size. 
        # If the batch size is bigger, it'll iterate through more data before it updates  - we get more 'average' values
        #print(history.history.keys())
        # "Loss"
        plt.plot(history.history['loss']) # loss on training data
        plt.plot(history.history['val_loss']) # loss on validation (test) data
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
        y_predict = y_predict[0]
        y_test = [y_test.iloc[i]['cost_to_load'] for i in range(y_test.shape[0])]
        maxval = np.max(np.maximum(y_predict,y_test))
        plt.scatter(y_predict,y_test, color = 'black')
        plt.plot(np.arange(0,maxval,100), np.arange(0,maxval,100), color = 'blue')
        plt.title('predicted (x) vs. actual (y)')
        plt.show()
        '''
        plt.scatter(x_data[['price_A']], y_data[['cost_to_load']])
        plt.title('price_A vs. cost_to_load')
        plt.show()
        
        plt.scatter(x_data[['price_B']], y_data[['cost_to_load']])
        plt.title('price_B vs. cost_to_load')
        plt.show()
        
        plt.scatter(x_data[['load']], y_data[['cost_to_load']])
        plt.title('load vs. cost_to_load')
        plt.show()
        
        plt.scatter(x_test[:,2], y_predict)
        plt.title('test load vs. cost_to_load predictions')
        plt.show()
        '''
    elif model == 'linear_regressor_full': # create, train, run linear regression model        
        
        # split into train and test sets
        training_examples = x_data.sample(n=21)
        train_idxs = training_examples.index
        training_targets = y_data.iloc[train_idxs]
        test_idxs = [x for x in x_data.index if x not in train_idxs]
        validation_examples = x_data.iloc[test_idxs]
        validation_targets = y_data.iloc[test_idxs]
        
        # NOTE: NOT SCALED
        # TENSORFLOW VERSION
        def construct_feature_columns(input_features):
            return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
        
        def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
            # convert df into dict of np arrays
            features = {key:np.array(value) for key, value in dict(features).items()}
            
            # construct dataset and configure batching/repeating
            ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit
            ds = ds.batch(batch_size).repeat(num_epochs)
            
            # shuffle
            if shuffle:
                ds = ds.shuffle(10000)
                
            # return next batch 
            features, labels = ds.make_one_shot_iterator().get_next()
            return features, labels
        
        def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
            periods = 10
            steps_per_period = steps/periods
            
            # create lin reg object
            my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
            linear_regressor = tf.estimator.LinearRegressor(\
                        feature_columns = construct_feature_columns(training_examples), \
                        optimizer = my_optimizer)
            
            # create input functions
            training_input_fn = lambda: my_input_fn(training_examples, \
                                        training_targets['cost_to_load'], \
                                        batch_size=batch_size)
            predict_training_input_fn = lambda: my_input_fn(training_examples, \
                                        training_targets['cost_to_load'], \
                                        num_epochs=1, \
                                        shuffle=False)
            predict_validation_input_fn = lambda: my_input_fn(validation_examples, \
                                        validation_targets['cost_to_load'], \
                                        num_epochs=1, \
                                        shuffle=False)
            
            # train the model
            training_rmse = []
            validation_rmse = []
            for period in range(0, periods):
                # train
                linear_regressor.train(\
                        input_fn = training_input_fn, steps=steps_per_period)
                training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
                training_predictions = np.array([item['predictions'][0] for item in training_predictions])
                
                validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
                validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
                
                plt.scatter(training_predictions, training_targets)
                plt.title('predictions vs. targets')
                plt.show()
                print(validation_predictions)
                print(validation_targets)
                
                training_rootmse = math.sqrt(mean_squared_error(training_predictions, training_targets))
                validation_rootmse = math.sqrt(mean_squared_error(validation_predictions, validation_targets))
                
                print(" period %02d : %0.2f" % (period, training_rootmse))
                
                training_rmse.append(training_rootmse)
                validation_rmse.append(validation_rootmse)
                
                
                # Output a graph of loss metrics over periods.
                plt.ylabel("RMSE")
                plt.xlabel("Periods")
                plt.title("Root Mean Squared Error vs. Periods")
                plt.tight_layout()
                plt.plot(training_rmse, label="training")
                plt.plot(validation_rmse, label="validation")
                plt.legend()
                plt.show()
                
                return linear_regressor
            
            
        train_model(\
                        learning_rate = 0.001, \
                        steps = 500, \
                        batch_size = 5, \
                        training_examples = training_examples, \
                        training_targets = training_targets, \
                        validation_examples = validation_examples, \
                        validation_targets = validation_targets)
            
        '''
        # SCIKITLEARN VERSION
        # create numpy arrays for simplicity
        x = x_data
        y = y_data
        
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.8)
        print(x_train)
        print(x_test)

        regr = linear_model.LinearRegression()
        regr.fit(x_train,y_train)
        #predict
        #y_pred = regr.predict(x_test.reshape(-1,1))
        y_pred = regr.predict(x_test)
        # plots
        print(y_pred)
        
        
        print('Coefficients: \n', regr.coef_)
        print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))
        print("Variance score: %.2f" % r2_score(y_test, y_pred))
        plt.scatter(y_test, y_pred, color='black',s=20)
        plt.plot(np.arange(0,np.max(y_pred)), np.arange(0,np.max(y_pred)),color='blue',linewidth=3)
        plt.title('actual cost to load vs. prediction: full lin. reg.')
        plt.xlabel('actual')
        plt.ylabel('prediction')
        plt.show()
        '''
        

    elif model == 'linear_regressor_simple': # create, train, run linear regression model with most highly correlated variable      
        # create numpy arrays for simplicity
        x = np.array(x_data[['load']])[:,0]
        y = np.array(y_data[['cost_to_load']])[:,0]
        
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.8)

        regr = linear_model.LinearRegression()
        regr.fit(x_train.reshape(-1,1),y_train)
        #predict
        y_pred = regr.predict(x_test.reshape(-1,1))
        # plots
        print(x_test.shape)
        
        
        print('Coefficients: \n', regr.coef_)
        print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))
        print("Variance score: %.2f" % r2_score(y_test, y_pred))
        plt.scatter(x_test, y_test, color='black',s=20)
        plt.plot(x_test,y_pred,color='blue',linewidth=3)
        plt.title('load vs. cost-to-load: simple lin. reg.')
        plt.xlabel('load')
        plt.ylabel('cost_to_load')
        plt.show()
        
    
    
    else:
        print("Selected model is not valid.")
        exit
        
        
        
run_model(model='neural_network')
'''
print(len(x_data[['price_A']]))
print(len(y_data[['cost_to_load']]))
plt.scatter(x_data[['price_A']], y_data[['cost_to_load']], s=3)
plt.title('price_A vs. cost_to_load')
plt.show()

plt.scatter(y_data[['cost_to_load']], y_data[['cost_to_load']], s=3)
plt.show()

print(np.corrcoef(np.array(x_data[['price_A']])[:,0],np.array(y_data[['cost_to_load']])[:,0]))
print(np.corrcoef(np.array(x_data[['price_B']])[:,0],np.array(y_data[['cost_to_load']])[:,0]))
print(np.corrcoef(np.array(x_data[['load']])[:,0],np.array(y_data[['cost_to_load']])[:,0]))
'''