# -*- coding: utf-8 -*-
"""
Connect to a PLEXOS Solution File, load data into pandas DataFrame,
and write an Excel file.

Created on Fri Sep 08 15:03:46 2017

@author: Steven
"""

# standard Python/SciPy libraries
import os
import pandas as pd
import sqlite3 as sql

# Python .NET interface
from dotnet.seamless import add_assemblies, load_assembly

# load PLEXOS assemblies... replace the path below with the installation
#   installation folder for your PLEXOS installation.
#add_assemblies('C:/Program Files (x86)/Energy Exemplar/PLEXOS 7.4/')
add_assemblies('C:/Program Files (x86)/Energy Exemplar/PLEXOS 8.0/')
load_assembly('PLEXOS7_NET.Core')
load_assembly('EEUTILITY')

# Import from .NET assemblies (both PLEXOS and system)
from PLEXOS7_NET.Core import *
from EEUTILITY.Enums import *

# Create a PLEXOS solution file object and load the solution
sol = Solution()
sol_file = 'Model Q2 Week1 DA Solution.zip' # replace with your solution file
if not os.path.exists(sol_file):
    print 'No such file'
    exit
    
sol.Connection(sol_file)

'''
Simple query: works similarly to PLEXOS Solution Viewer

Solution.Query(phase, collection, parent, child, period, series, props)
    phase -> SimulationPhaseEnum
    collection -> CollectionEnum
    parent -> the name of a parent object or ''
    child -> the name of a child object or ''
    period -> PeriodEnum
    series -> SeriesTypeEnum
    props -> a string containing an integer indicating the Property to query or ''
returns a ADODB recordset... however, you don't *need* to worry about that...
'''

# Run the query
results = sol.Query(SimulationPhaseEnum.STSchedule, \
                    CollectionEnum.SystemGenerators, \
                    '', \
                    '', \
                    PeriodEnum.FiscalYear, \
                    SeriesTypeEnum.Values, \
                    '')

# Check to see if the query had results
if results.EOF:
    print 'No results'
    exit

# Create a DataFrame with a column for each column in the results
df = pd.DataFrame(columns=[x.Name for x in results.Fields])

# loop through the recordset
idx = 0    
while not results.EOF:
    df.loc[idx] = [str(x.Value) for x in results.Fields]
    idx += 1
    results.MoveNext() #VERY IMPORTANT

# push results to a SQLite3 database    
conn = sql.connect('plexos.db')
df.to_sql('MyQuery', conn, if_exists='replace') # 'MyQuery' is the name of the table

# try a query and push results to an Excel file
sql_df = pd.read_sql_query('SELECT * FROM MyQuery WHERE property_name = "Fuel Cost";', conn)
wb = pd.ExcelWriter('sql_query.xlsx')
sql_df.to_excel(wb, 'SQL Query')

# save the excel file
wb.save()

# close the sqlite3 connection
conn.close()
