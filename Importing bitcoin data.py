# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:41:44 2022

@author: marti
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
page=requests.get('https://gz.blockchair.com/bitcoin/transactions/')
data=page.text
soup=bs(data)
links=[]
for link in soup.find_all('a'):
    links.append('https://gz.blockchair.com/bitcoin/transactions/' + link.get('href'))
links = links[4:-8]
links[0:5]

import urllib.request
import gzip

# insert loop here for multiple URLs

url = links[-1]
print(url)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
with urllib.request.urlopen(url) as response:
    with gzip.GzipFile(fileobj=response) as uncompressed:
        file_header = uncompressed.read()

from io import StringIO
# Create Dataframe from first object
s = str(file_header, 'utf-8')
data = StringIO(s)
df=pd.read_csv(data, sep = '\t')
print(df.tail())
import numpy as np
np.shape(df)

import matplotlib.pyplot as plt
plt.scatter(df['size'], df['fee'], s=2)
plt.xlabel("SIZE")
plt.ylabel("FEE")
plt.title("Fee Versus Transaction Size")
plt.show()

plt.plot(df['time'],df['fee'])
plt.xlabel('TIME')
plt.ylabel('FEE')
plt.title('Fees Paid over the course of the Day')

#################################################
# Define the variables required for the master_df
#################################################
Time_Stamp = df['time']
Today_Time_Stamp = Time_Stamp[1]
Date = Today_Time_Stamp[0:10]

Number_of_Transactions = len(df)
Number_of_Coinbase_Transactions = len(df[(df['is_coinbase']==1)])
Coinbase_Transactions_Output_Total = df.loc[df['is_coinbase'] == 1, 'output_total'].sum()

Total_Size_of_Transactions = np.sum(df['size'])
Mean_Size_of_Transactions = np.mean(df['size'])
Median_Size_of_Transactions = np.median(df['size'])
StdDev_Size_of_Transactions = np.std(df['size'])

Total_Weight_of_Transactions = np.sum(df['weight'])
Mean_Weight_of_Transactions = np.mean(df['weight'])
Median_Weight_of_Transactions = np.median(df['weight'])
StdDev_Weight_of_Transactions = np.std(df['weight'])

Total_Input_Count = np.sum(df['input_count'])
Mean_Input_Count = np.mean(df['input_count'])
Median_Input_Count = np.median(df['input_count'])
StdDev_Input_Count = np.std(df['input_count'])

Total_Output_Count = np.sum(df['output_count'])
Mean_Output_Count = np.mean(df['output_count'])
Median_Output_Count = np.median(df['output_count'])
StdDev_Output_Count = np.std(df['output_count'])

Total_Input_Total = np.sum(df['input_total'])
Mean_Input_Total = np.mean(df['input_total'])
Median_Input_Total = np.median(df['input_total'])
StdDev_Input_Total = np.std(df['input_total'])

Total_Output_Total = np.sum(df['output_total'])
Mean_Output_Total = np.mean(df['output_total'])
Median_Output_Total = np.median(df['output_total'])
StdDev_Output_Total = np.std(df['output_total'])

Total_Fees = np.sum(df['fee'])
Mean_Fees = np.mean(df['fee'])
Median_Fees = np.median(df['fee'])
StdDev_Fees = np.std(df['fee'])
Fee_per_KB = Total_Fees/(Total_Size_of_Transactions/1000)

#########################################################
# Create the master_df1
#########################################################
d = {'Date':[], 'Number_of_Transactions':[], 'Number_of_Coinbase_Transactions':[], 
     'Total_Size_of_Transactions':[], 'Mean_Size_of_Transactions':[], 'Median_Size_of_Transactions':[], 'StdDev_Size_of_Transactions':[], 
     'Total_Weight_of_Transactions':[], 'Mean_Weight_of_Transactions':[], 'Median_Weight_of_Transactions':[], 'StdDev_Weight_of_Transactions':[], 
     'Total_Input_Count':[], 'Mean_Input_Count':[], 'Median_Input_Count':[], 'StdDev_Input_Count':[], 
     'Total_Output_Count':[], 'Mean_Output_Count':[], 'Median_Output_Count':[], 'StdDev_Output_Count':[],
     'Total_Input_Total':[], 'Mean_Input_Total':[], 'Median_Input_Total':[], 'StdDev_Input_Total':[], 
     'Total_Output_Total':[], 'Mean_Output_Total':[], 'Median_Output_Total':[], 'StdDev_Output_Total':[], 
     'Total_Fees':[], 'Mean_Fees':[], 'Median_Fees':[], 'StdDev_Fees':[], 'Fee_per_KB'}
master_df1 = pd.DataFrame(data=d)
# print(master_df1)

# Append a row to master_df1
new_row = {'Date': Date, 'Number_of_Transactions': Number_of_Transactions, 'Number_of_Coinbase_Transactions':Number_of_Coinbase_Transactions, 
           'Total_Size_of_Transactions':Total_Size_of_Transactions, 'Mean_Size_of_Transactions':Mean_Size_of_Transactions, 
           'Median_Size_of_Transactions':Median_Size_of_Transactions, 'StdDev_Size_of_Transactions':StdDev_Size_of_Transactions
           'Total_Weight_of_Transactions':Total_Weight_of_Transactions, 'Mean_Weight_of_Transactions':Mean_Weight_of_Transactions, 
           'Median_Weight_of_Transactions':Median_Weight_of_Transactions, 'StdDev_Weight_of_Transactions':StdDev_Weight_of_Transactions
           'Total_Input_Count':Total_Input_Count, 'Mean_Input_Count':Mean_Input_Count, 
           'Median_Input_Count':Median_Input_Count, 'StdDev_Input_Count':StdDev_Input_Count
           'Total_Output_Count':Total_Output_Count, 'Mean_Output_Count':Mean_Output_Count, 
           'Median_Output_Count':Median_Output_Count, 'StdDev_Output_Count':StdDev_Output_Count
           'Total_Input_Total':Total_Input_Total, 'Mean_Input_Total':Mean_Input_Total, 
           'Median_Input_Total':Median_Input_Total, 'StdDev_Input_Total':StdDev_Input_Total
           'Total_Output_Total':Total_Output_Total, 'Mean_Output_Total':Mean_Output_Total, 
           'Median_Output_Total':Median_Output_Total, 'StdDev_Output_Total':StdDev_Output_Total
           'Total_Fees':Total_Fees, 'Mean_Fees':Mean_Fees, 
           'Median_Fees':Median_Fees, 'StdDev_Fees':StdDev_Fees}
#append row to the dataframe
master_df1 = master_df1.append(new_row, ignore_index=True)

print('\n\nNew row added to DataFrame\n--------------------------')
print(master_df1)
# master_df.drop([0])
