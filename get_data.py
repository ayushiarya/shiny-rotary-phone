"""
GET DATA FROM SOLUTION FILES IN A FOLDER SYSTEM TO SQLite TUNING DB

"""

import os
import re
import pandas as pd
import sqlite3 as sql
import json
import ast
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000000000)

# Python .NET interface
from dotnet import add_assemblies, load_assembly
from dotnet.overrides import type, isinstance, issubclass

# load PLEXOS assemblies... replace the path below with the installation
#   installation folder for your PLEXOS installation.
add_assemblies('C:/Program Files (x86)/Energy Exemplar/PLEXOS 8.0/')

try:
    load_assembly('PLEXOS7_NET.Core')
    load_assembly('EEUTILITY')
except:
    print("\nException caught \n")

# Import from .NET assemblies (both PLEXOS and system)
from PLEXOS7_NET.Core import *
from EEUTILITY.Enums import *
from System import *


# helper functions
def convert(jobj, key, query_num=0):
    '''
    Function to convert strings of enumeration parameter to correct type
    Inputs:
            jobj (dictionary) : dictionary representation of json object
            key (string) : key in 'QueryParams' in metadata_dict
            querynum : nonnegative integer representing query number
    Outputs:
            corresponding converted object (string, boolean, enum, etc.)
    '''
    if key in ["strCSVFile", "ParentName", "ChildName", "PropertyList", \
               "TimesliceList", "SampleList", "ModelName", "Category", \
               "Separator"]:
        return str(jobj['QueryParams'][query_num][key])
    elif key == "bAppendToFile":
        return ast.literal_eval(str(jobj['QueryParams'][query_num][key]))
    elif key == "SimulationPhaseId":
        return Enum.Parse(type(SimulationPhaseEnum), str(jobj['QueryParams'][query_num][key]))
    elif key == "CollectionId" : 
        return Enum.Parse(type(CollectionEnum), str(jobj['QueryParams'][query_num][key]))
    elif key == "PeriodTypeId":
        return Enum.Parse(type(PeriodEnum), str(jobj['QueryParams'][query_num][key]))
    elif key == "SeriesTypeId":
        return Enum.Parse(type(SeriesTypeEnum), str(jobj['QueryParams'][query_num][key]))
    elif key == "AggregationType":
        return Enum.Parse(type(AggregationEnum), str(jobj['QueryParams'][query_num][key]))
    elif key == "DateFrom" or key == "DateTo":
        return DateTime.Parse(str(jobj['QueryParams'][query_num][key]))
    else:
        print("ERROR: Invalid key parameter supplied.")
        exit

def get_params(json_dict):
    '''
    Function to get parameters for query from a dictionary
    Inputs:
            dict (dictionary) : dictionary representation of json object
    Outputs:
            list of tuples of parameters for QueryToCSV method
    '''
    params = []

    keys = ['strCSVFile', 'bAppendToFile', 'SimulationPhaseId', 
            'CollectionId', 'ParentName', 'ChildName', 'PeriodTypeId', \
            'SeriesTypeId', 'PropertyList', 'DateFrom', 'DateTo', \
            'TimesliceList', 'SampleList', 'ModelName', \
            'AggregationType', 'Category', 'Separator']
    for j in range(len(json_dict['QueryParams'])):
            params.append(tuple([convert(json_dict, keys[i], j) for i in range(len(keys))]))
    return params


tuning_df = pd.DataFrame(columns=['tuning_key','source','X'])
idx = 0
pattern = re.compile("^Model.*Solution\.zip")
string1 = "Model "
string2 = " Solution.zip"

# load metadata from .json file and loop over json objects
json_objects = json.loads(open('jsontest.json', "r").read())
for i in range(len(json_objects)):
    jobj = json_objects[i]
    # get info from json metadata
    params_list = get_params(jobj)
    tuning_key = str(jobj['ModelName'])+str(jobj['Y']) # e.g. SMUDFuelCost
    # Traverse a file tree in search of solution files from which to pull data
    
    for dirName, subDirs, files in os.walk(str(jobj['RootFolder'])):
        for filename in files:
            df = pd.DataFrame()
            source = 'temp'
            if bool(pattern.match(filename)): # if True, then "Model ___ Solution.zip" pattern matched
                # create source: folder name / model name
                tempsource = filename.replace(string1,'')
                source = os.path.basename(dirName) + "\\" + tempsource.replace(string2,'') 
                # A. Connect
                # Create a PLEXOS solution file object and load the solution
                sol = Solution()
                sol_file = dirName + "\\" + filename # replace with your solution file
                if not os.path.exists(sol_file):
                    print ('No such file')
                    exit
                sol.Connection(sol_file)
                # B. Pull data (query to Pandas via CSV)
                
                # a. Alias the Query method with the arguments you plan to use.
                # Set up the query
                query = sol.QueryToCSV[String, Boolean, SimulationPhaseEnum, \
                                       CollectionEnum, String, String, \
                                       PeriodEnum, SeriesTypeEnum, String, \
                                       Object, Object, String, String, \
                                       String, AggregationEnum, String, String]
                for k in range(len(params_list)):
                    
                    params = params_list[k] # construct tuple to send as parameters
                    # c. Use the __invoke__ method of the alias to call the query method.
                    results = query.__invoke__(params)
                    '''
                    params = params_list[k][:8]
                    results = sol.QueryToCSV(*params_list[k][:8])
                    '''
    
                    if k == 0:
                        df = pd.read_csv(params[0]) # create dataframe with query results
                    else:
                        df2 = pd.read_csv(params[0]) # concatenate to existing
                        df = pd.concat([df,df2])
                # do not leave NaN values - will create problem later when extracting data back out
                df = df.fillna("None")
                X = str(df)
                Y = str(jobj['Y']) # will come from metadata
                tuning_df.loc[idx] = [tuning_key, source, X]
                idx += 1
# C. Push tuning data to a db (SQLite) with source information * ONE TIME *
conn = sql.connect('TuningDB.db')
tuning_df.to_sql('TuningData', conn, if_exists='replace') # 'TuningData' is the name of the table
# close the SQLite3 connection
conn.close()

print("Data successfully retrieved.")