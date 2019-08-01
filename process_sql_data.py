"""
GET "X" DATA FROM TUNING DB (SQLite3) INTO A PANDAS DATAFRAME.
SAVES DATAFRAME TO 'x_data.csv'

"""

import sqlite3 as sql
import pandas as pd
import re
import numpy as np


def get_X(df, xcols):
    '''
    Returns row values from columns specified in xcols.
    Use to get 'x 'data from tuning DB.
    
    Inputs:     df - pandas dataframe of tuning data, created by get_data.py
                xcols -list of column name of interest, e.g. ['value']
    Outputs:    pandas dataframe with "x data" (as specified by xcols)
    '''
    mystr = str(df['X'][0]).strip()
    col_list = re.sub("[^\w]", " ",  mystr).split()[:24] # get column names

    new_df = pd.DataFrame(columns=col_list) # DataFrame to hold results

    # loop over rows of tuning DB - each containts a blob of data
    for i in range(len(df)):
        
        mystr = str(df['X'][i]).strip()
        # remove column names (everything up to last column name, "phase_name")
        mystr = re.sub(r"[a-zA-Z_\s]+phase_name", "", mystr)
        mystr =re.sub("\n","",mystr) # remove newline characters
        data_list = re.split(r'\s{2,}', mystr) # split on multiple whitespaces
        # remove index leftover from DataFrame construction in get_data.py
        data_list = data_list[1:]
        data_list = np.reshape(data_list,(int(len(data_list)/24),24))
        # create temporary df to hold data from current row of tuning db
        temp_df = pd.DataFrame(columns = col_list, data = data_list)
        # append to existing df holding all x data thus far
        new_df = pd.concat([new_df, temp_df])
        
    new_df.reset_index(drop=True, inplace=True) # fixes an indexing problem
    
    if 'value' in xcols:
        new_df['value'] = pd.to_numeric(new_df['value']) # string --> float

    if xcols == ['all']:
        return new_df
    else:
        return new_df[xcols]
    
def get_property_names(df, property_names):
    return df[df['property_name'].isin(property_names)]

# Example of how it works
conn = sql.connect("TuningDB.db") # connect to the Tuning DB
tuning_df = pd.read_sql_query("SELECT * from TuningData", conn)

#x_df = get_X(tuning_df, \
#             ['model_name', 'collection_name', 'property_name', 'property_id', \
#              'child_name', 'value', 'unit_name'])
x_df = get_X(tuning_df, ['model_name','collection_id','collection_name', \
                         'category_id','child_id','child_name', \
                         'property_id','property_name','band_id','value', \
                         'unit_id','unit_name'])
    
#x_df = get_property_names(x_df, ['Cost to Load'])

conn.close() # close the connection to SQL

x_df.to_csv('x_data.csv')
print("DataFrame successfully created")
