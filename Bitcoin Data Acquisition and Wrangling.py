# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:41:44 2022

@author: marti
"""
#######################################################################
# Import Libraries
#######################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import gzip
import ssl
from io import StringIO
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime

######################################################################
# Acquire list of URLs from Blockchair
######################################################################
page=requests.get('https://gz.blockchair.com/bitcoin/transactions/')
data=page.text
soup=bs(data)
links=[]
for link in soup.find_all('a'):
    links.append('https://gz.blockchair.com/bitcoin/transactions/' + link.get('href'))   
# print(links)    
links = links[4:-8]
links = links[-145:-121]
print(links)
########################################################################
# For last URL in links, write contents to dataframe df
########################################################################
url = links[-1]
ssl._create_default_https_context = ssl._create_unverified_context
with urllib.request.urlopen(url) as response:
    with gzip.GzipFile(fileobj=response) as uncompressed:
        file_header = uncompressed.read()

s = str(file_header, 'utf-8') # Create Dataframe from first object
data = StringIO(s)
df=pd.read_csv(data, sep = '\t')

#################################################
# Define the variables required for the master_df3
#################################################
Time_Stamp = df['time']
Today_Time_Stamp = Time_Stamp[1]
Today_Time_Stamp = Today_Time_Stamp[0:10]
Date = datetime.strptime(Today_Time_Stamp, '%Y-%m-%d')

Number_of_Transactions = len(df)
Number_of_Coinbase_Transactions = len(df[(df['is_coinbase']==1)])
Coinbase_Transactions_Output_Total = df.loc[df['is_coinbase'] == 1, 'output_total'].sum()
Number_of_Zero_Fee_Transactions = len(df[(df['fee']==0)])

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
# Create the master_df3
#########################################################
d = {'Date':[], 'Number_of_Transactions':[], 'Number_of_Coinbase_Transactions':[], 'Coinbase_Transactions_Output_Total':[], 'Number_of_Zero_Fee_Transactions':[], 
     'Total_Size_of_Transactions':[], 'Mean_Size_of_Transactions':[], 'Median_Size_of_Transactions':[], 'StdDev_Size_of_Transactions':[], 
     'Total_Weight_of_Transactions':[], 'Mean_Weight_of_Transactions':[], 'Median_Weight_of_Transactions':[], 'StdDev_Weight_of_Transactions':[], 
     'Total_Input_Count':[], 'Mean_Input_Count':[], 'Median_Input_Count':[], 'StdDev_Input_Count':[], 
     'Total_Output_Count':[], 'Mean_Output_Count':[], 'Median_Output_Count':[], 'StdDev_Output_Count':[],
     'Total_Input_Total':[], 'Mean_Input_Total':[], 'Median_Input_Total':[], 'StdDev_Input_Total':[], 
     'Total_Output_Total':[], 'Mean_Output_Total':[], 'Median_Output_Total':[], 'StdDev_Output_Total':[], 
     'Total_Fees':[], 'Mean_Fees':[], 'Median_Fees':[], 'StdDev_Fees':[], 'Fee_per_KB':[]}
master_df3 = pd.DataFrame(data=d)

# Append first row to master_df2
new_row = {'Date': Date, 'Number_of_Transactions': Number_of_Transactions, 'Number_of_Coinbase_Transactions':Number_of_Coinbase_Transactions,
           'Coinbase_Transactions_Output_Total':Coinbase_Transactions_Output_Total, 'Number_of_Zero_Fee_Transactions':Number_of_Zero_Fee_Transactions,
           'Total_Size_of_Transactions':Total_Size_of_Transactions, 'Mean_Size_of_Transactions':Mean_Size_of_Transactions, 
           'Median_Size_of_Transactions':Median_Size_of_Transactions, 'StdDev_Size_of_Transactions':StdDev_Size_of_Transactions,
           'Total_Weight_of_Transactions':Total_Weight_of_Transactions, 'Mean_Weight_of_Transactions':Mean_Weight_of_Transactions, 
           'Median_Weight_of_Transactions':Median_Weight_of_Transactions, 'StdDev_Weight_of_Transactions':StdDev_Weight_of_Transactions,
           'Total_Input_Count':Total_Input_Count, 'Mean_Input_Count':Mean_Input_Count, 
           'Median_Input_Count':Median_Input_Count, 'StdDev_Input_Count':StdDev_Input_Count,
           'Total_Output_Count':Total_Output_Count, 'Mean_Output_Count':Mean_Output_Count, 
           'Median_Output_Count':Median_Output_Count, 'StdDev_Output_Count':StdDev_Output_Count,
           'Total_Input_Total':Total_Input_Total, 'Mean_Input_Total':Mean_Input_Total, 
           'Median_Input_Total':Median_Input_Total, 'StdDev_Input_Total':StdDev_Input_Total,
           'Total_Output_Total':Total_Output_Total, 'Mean_Output_Total':Mean_Output_Total, 
           'Median_Output_Total':Median_Output_Total, 'StdDev_Output_Total':StdDev_Output_Total,
           'Total_Fees':Total_Fees, 'Mean_Fees':Mean_Fees, 
           'Median_Fees':Median_Fees, 'StdDev_Fees':StdDev_Fees, 'Fee_per_KB':Fee_per_KB}
master_df3 = master_df3.append(new_row, ignore_index=True) #append row to the dataframe

##########################################################################################################
# For all remaining URLs, write the bitcoin transactions to dataframe df, 
# then write a summary of these as a single row to master_df3
##########################################################################################################
links = links[10:-1]
print(links)
for next_url in links:
    ssl._create_default_https_context = ssl._create_unverified_context
    with urllib.request.urlopen(next_url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_header = uncompressed.read()

    s = str(file_header, 'utf-8') # Create Dataframe from first object
    data = StringIO(s)
    df=pd.read_csv(data, sep = '\t')
    #################################################
    # Recalculate variables
    #################################################
    Time_Stamp = df['time']
    Today_Time_Stamp = Time_Stamp[1]
    Today_Time_Stamp = Today_Time_Stamp[0:10]
    Date = datetime.strptime(Today_Time_Stamp, '%Y-%m-%d')

    Number_of_Transactions = len(df)
    Number_of_Coinbase_Transactions = len(df[(df['is_coinbase']==1)])
    Coinbase_Transactions_Output_Total = df.loc[df['is_coinbase'] == 1, 'output_total'].sum()
    Number_of_Zero_Fee_Transactions = len(df[(df['fee']==0)])

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
    
    # Append next row to master_df3
    new_row = {'Date': Date, 'Number_of_Transactions': Number_of_Transactions, 'Number_of_Coinbase_Transactions':Number_of_Coinbase_Transactions, 
               'Coinbase_Transactions_Output_Total':Coinbase_Transactions_Output_Total, 'Number_of_Zero_Fee_Transactions':Number_of_Zero_Fee_Transactions,
               'Total_Size_of_Transactions':Total_Size_of_Transactions, 'Mean_Size_of_Transactions':Mean_Size_of_Transactions, 
               'Median_Size_of_Transactions':Median_Size_of_Transactions, 'StdDev_Size_of_Transactions':StdDev_Size_of_Transactions,
               'Total_Weight_of_Transactions':Total_Weight_of_Transactions, 'Mean_Weight_of_Transactions':Mean_Weight_of_Transactions, 
               'Median_Weight_of_Transactions':Median_Weight_of_Transactions, 'StdDev_Weight_of_Transactions':StdDev_Weight_of_Transactions,
               'Total_Input_Count':Total_Input_Count, 'Mean_Input_Count':Mean_Input_Count, 
               'Median_Input_Count':Median_Input_Count, 'StdDev_Input_Count':StdDev_Input_Count,
               'Total_Output_Count':Total_Output_Count, 'Mean_Output_Count':Mean_Output_Count, 
               'Median_Output_Count':Median_Output_Count, 'StdDev_Output_Count':StdDev_Output_Count,
               'Total_Input_Total':Total_Input_Total, 'Mean_Input_Total':Mean_Input_Total, 
               'Median_Input_Total':Median_Input_Total, 'StdDev_Input_Total':StdDev_Input_Total,
               'Total_Output_Total':Total_Output_Total, 'Mean_Output_Total':Mean_Output_Total, 
               'Median_Output_Total':Median_Output_Total, 'StdDev_Output_Total':StdDev_Output_Total,
               'Total_Fees':Total_Fees, 'Mean_Fees':Mean_Fees, 
               'Median_Fees':Median_Fees, 'StdDev_Fees':StdDev_Fees, 'Fee_per_KB':Fee_per_KB}
    master_df3 = master_df3.append(new_row, ignore_index=True) #append row to the master_df3 dataframe
    
master_df3.to_csv(r'C:\Users\marti\Desktop\master_df3_2.csv', index = False, header=True)

master_df3=pd.read_csv(r'C:\Users\marti\Desktop\master2.csv')
print(master_df3.head())
master_df3.sort_values(by=['Date'])  
print(master_df3.head())  
print(type(master_df3['Date']))
master_df3.set_index('Date')

# plt.scatter(x, y, kwargs)
plt.scatter(master_df3['Mean_Size_of_Transactions'], master_df3['Mean_Fees'], s=2)
plt.xlabel("Mean_Size_of_Transactions")
plt.ylabel("Mean_Fees")
plt.title("Mean Fees Vs Mean_Size_of_Transactions")
plt.show()

master_df3.corr()

sns.lmplot(x='Mean_Size_of_Transactions',y='Mean_Fees',data=master_df3)
plt.xlabel('Mean_Size_of_Transactions')
plt.ylabel('Mean_Fees')
plt.title("Mean Fees Vs Mean_Size_of_Transactions")
plt.show()

reduced_master_df3 = master_df3[['Number_of_Transactions', 'Number_of_Zero_Fee_Transactions', 'Mean_Size_of_Transactions', 'Mean_Input_Count', 'Mean_Output_Count', 'Mean_Input_Total', 'Mean_Output_Total', 'Mean_Fees']]
sns.heatmap(reduced_master_df3.corr(), annot=True, cmap='Blues', fmt='.3f')

sns.pairplot(reduced_master_df3)

# Models
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Drawing a regression Line
X = reduced_master_df3[['Mean_Size_of_Transactions', 'Mean_Input_Count']] 
Y = reduced_master_df3[['Mean_Fees']]
# reduced_master_df3 = reduced_master_df3[:317]
reg = LinearRegression().fit(X, Y)
reg.predict([[750,2]])
reg.score(X, Y)

# Evaluation Using Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
reg = LinearRegression().fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

mean_squared_error(Y_test, Y_pred)**0.5

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)
print(tree.export_text(clf))

# Making Predictions
Y_pred = clf.predict(X_test)
accuracy_score(Y_test, Y_pred)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

np.isnan(reduced_master_df3['Mean_Fees'].any())
np.isfinite(reduced_master_df3['Mean_Fees'].all())
print(type(X))
print(reduced_master_df3.head())

plt.plot(df['time'],df['fee'])
plt.xlabel('TIME')
plt.ylabel('FEE')
plt.title('Fees Paid over the course of the Day')
