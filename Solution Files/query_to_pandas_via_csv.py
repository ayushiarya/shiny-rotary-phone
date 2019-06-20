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
from datetime import datetime


# Python .NET interface
from dotnet.seamless import add_assemblies, load_assembly

# load PLEXOS assemblies... replace the path below with the installation
#   installation folder for your PLEXOS installation.
add_assemblies('C:/Program Files (x86)/Energy Exemplar/PLEXOS 7.4/')
load_assembly('PLEXOS7_NET.Core')
load_assembly('EEUTILITY')

# Import from .NET assemblies (both PLEXOS and system)
from PLEXOS7_NET.Core import *
from EEUTILITY.Enums import *

# Create a PLEXOS solution file object and load the solution
sol = Solution()
sol_file = 'Model Q2 Week1 DA Solution.zip' # replace with your solution file
csv_file = 'generator_data.csv'

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
results = sol.QueryToCSV( \
                    csv_file, False,
                    SimulationPhaseEnum.STSchedule, \
                    CollectionEnum.SystemGenerators, \
                    '', \
                    '', \
                    PeriodEnum.FiscalYear, \
                    SeriesTypeEnum.Values, \
                    '')

df = pd.read_csv(csv_file)
    
wb = pd.ExcelWriter('query.xlsx')
df.to_excel(wb, 'Query') # 'Query' is the name of the worksheet
wb.save()
