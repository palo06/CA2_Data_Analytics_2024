# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:22:24 2024

@author: maria
"""

# Uploading and transforming the CSV file to JSON File
import csv
import json
from pymongo import MongoClient
import pymongo
import pandas as pd
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


def csv_to_json(csv_file_path):
    data = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return json.dumps(data, indent=4)

csv_file_path = 'AveragePrice.csv'
json_data = csv_to_json(csv_file_path)

with open('AveragePrice.json', 'w') as jsonfile:
    jsonfile.write(json_data)

print("Conversion completed. JSON file saved as 'AveragePrice.json'")



#Storing the databse in MongoDB

def store_in_mongodb(json_file_path, mongodb_uri, db_name, collection_name):
    # Load the JSON data
    with open(json_file_path) as jsonfile:
        data = json.load(jsonfile)

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)

    # Access the database
    db = client[db_name]

    # Access the collection
    collection = db[collection_name]

    # Insert the data into the collection
    collection.insert_many(data)

    print("Data stored in MongoDB.")

# Provide the path to your JSON file, MongoDB URI, database name, and collection name
json_file_path = 'AveragePrice.json'
mongodb_uri = 'mongodb://localhost:27017'
db_name = 'StoredAveragePrice'
collection_name = 'averageprice_collection'

store_in_mongodb(json_file_path, mongodb_uri, db_name, collection_name)



# retriving the data from MongoDB and creating a structured dataset

def retrieve_data_from_mongodb(mongodb_uri, db_name, collection_name):
    # Connect to MongoDB
    client = pymongo.MongoClient(mongodb_uri)
    
    # Access the database
    db = client[db_name]
    
    # Access the collection
    collection = db[collection_name]
    
    # Retrieve the data
    data = list(collection.find({}))
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    
    return df

# Provide the MongoDB URI, database name, and collection name
mongodb_uri = 'mongodb://localhost:27017'
db_name = 'StoredAveragePrice'
collection_name = 'averageprice_collection'

# Retrieve the data from MongoDB and create a structured dataset
df = retrieve_data_from_mongodb(mongodb_uri, db_name, collection_name)



#2.Exploring the dataset
df.info()
print(df.describe())
print(df.head(5))
print('This dataset has {} observations with {} features.'.format(df.shape[0], df.shape[1]))
#Total number records by qualitative feature 
print("The total number of records by Statistic Label is: ",df.groupby("Statistic Label")["Statistic Label"].count(), sep="\n")
print("The total number of records by TLIST(Q1) is: ",df.groupby("TLIST(Q1)")["TLIST(Q1)"].count(), sep="\n")
print("The total number of records by Quarter is: ",df.groupby("Quarter")["Quarter"].count(), sep="\n")
print("The total number of records by C02343V02817  is: ",df.groupby("C02343V02817")["C02343V02817"].count(), sep="\n")
print("The total number of records by Area is: ",df.groupby("Area")["Area"].count(), sep="\n")
print("The total number of records by UNIT is: ",df.groupby("UNIT")["UNIT"].count(), sep="\n")
print("The total number of records  by VALUE is: ",df.groupby("VALUE")["VALUE"].count(), sep="\n")

# Split the last digit in column "TLIST(Q1)" and store it in a new column "Q"
df['Q'] = df['TLIST(Q1)'].astype(str).str[-1].astype(int)

# Store the digits without the last digit in a column "Year"
df['Year'] = df['TLIST(Q1)'].astype(str).str[:-1].astype(int)

# Drop the specified columns
columns_to_drop = ["UNIT", "TLIST(Q1)", "Quarter", "C02343V02817"]
df = df.drop(columns=columns_to_drop)



# Convert the "VALUE" column to integer, handling errors by coercing invalid values to NaN
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce').astype('Int64')

# Print a summary of the DataFrame
print(df.info())
print(df.describe())

#Dealing with missing values
print("The total number of non-null rows (by column) is:",df.notnull().sum(), sep="\n") 
print("The total number of null values (by column) is:" ,df.isnull().sum(), sep="\n")
print("The total number of null values for all the columns is:" ,df.isnull().sum().sum(), sep="\n")
print("The total number of missing values per colum in % is:",df.isnull().sum()/len(df)*100, sep="\n")
# Dropping rows with missing values 
df=df.dropna(subset=["VALUE"]) 
#New data shape
print(df.shape)

#Renaming variables
df=df.rename(columns={"Statistic Label":"House_Type"})

#Numerical
df.hist(column=["VALUE","Q","Year"] )

#Not numerical
fig = px.histogram(df, x="Statistic Label",color="Statistic Label")
fig.update_layout(
    title_text="Statistic Label", # title of plot
    xaxis_title_text="Statistic Label", # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, 
    bargroupgap=0.1
)
fig.show()
fig = px.histogram(df, x="Area",color="Area")
fig.update_layout(
    title_text="Area", # title of plot
    xaxis_title_text="Area", # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, 
    bargroupgap=0.1
)
fig.show()