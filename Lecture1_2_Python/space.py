# -*- coding: utf-8 -*-

# 1. importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Getting data
data = pd.read_csv('space_corrected.csv')
# plt.plot(data['Status Mission'])
# plt.show()

mask = data['Status Mission']=='Success'
failureMask = ~mask
print('Successes',len(mask[mask])) #length
print('Failures',len(failureMask[failureMask])) #length

subdata = data[failureMask] #dataframe= excel sheet
floridaCount=subdata['Location'].str.contains('Florida').sum()
print(floridaCount)

from datetime import datetime 

"""
subdata = data[failureMask]['Location']  #here change at end of class
subdata = list(subdata)
countFlorida = 0
countKazakhstan = 0
for location in subdata:
    if 'Florida' in location:
        countFlorida = countFlorida + 1
        print(location, countFlorida)
    if 'Kazakhstan' in location:
        countKazakhstan = countKazakhstan + 1
        print(location, countKazakhstan)
"""
