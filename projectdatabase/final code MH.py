# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:23:18 2024

@author: maria
@author: angeliki
@author: Paloma
"""

import csv
import json
from pymongo import MongoClient
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
import sqlalchemy
import os
import psycopg2
from psycopg2 import Error
import plotly.graph_objects as go
import requests


# PostgreSQL universal credentials
username = 'postgres'
password = '12345678'
host = 'localhost'
port = '5432'
database = 'postgres'

# MondoDB universal credentials
mongodb_uri = 'mongodb://localhost:27017'
mongodb_name = ''

#Code for Earnings database
def pull_data():
    url = "https://ws.cso.ie/public/api.jsonrpc?data=%7B%22jsonrpc%22:%222.0%22," \
          "%22method%22:%22PxStat.Data.Cube_API.ReadDataset%22,%22params%22:%7B%22class%22:%22query%22," \
          "%22id%22:%5B%5D,%22dimension%22:%7B%7D,%22extension%22:%7B%22pivot%22:null,%22codes%22:false," \
          "%22language%22:%7B%22code%22:%22en%22%7D,%22format%22:%7B%22type%22:%22JSON-stat%22," \
          "%22version%22:%222.0%22%7D,%22matrix%22:%22EHQ15%22%7D,%22version%22:%222.0%22%7D%7D "

    response = requests.get(url)

    if response.status_code == 200:
        # File content is stored in response.text
        file_content = response.text
        json_data = json.loads(file_content)
        return json_data
    else:
        print("Failed to fetch the file:", response.status_code)
        return 0

def structured_data(data: str):
    value_dictionary_list = []

    # pulling earnings list
    temp_data = data['result']['dimension']['STATISTIC']['category']['label']
    earnings_category_list = []
    earnings_category_dict_list = []

    for keys, values in temp_data.items():
        earnings_category_list.append(values)
        temp_dict = {
            keys: values
        }
        earnings_category_dict_list.append(temp_dict)

    # pulling quarter list
    temp_data = data['result']['dimension']['TLIST(Q1)']['category']['label']
    querters_list = []
    quarters_dict_list = []

    for keys, values in temp_data.items():
        querters_list.append(values)
        temp_dict = {
            keys: values
        }
        quarters_dict_list.append(temp_dict)

    # pulling economic sector list
    temp_data = data['result']['dimension']['C02665V03225']['category']['label']
    sector_list = []
    sector_dict_list = []

    for keys, values in temp_data.items():
        sector_list.append(values)
        temp_dict = {
            keys: values
        }
        sector_dict_list.append(temp_dict)

    # pulling value list
    value_list = data['result']['value']

    count = 0
    for each_earning in earnings_category_list:
        for each_quarter in querters_list:
            for each_sector in sector_list:
                temp_dict = {
                    "earning_category": each_earning,
                    "quarter": each_quarter,
                    "economic_sector": each_sector,
                    "value": value_list[count],
                }
                value_dictionary_list.append(temp_dict)
                count += 1

    return value_dictionary_list, earnings_category_dict_list, quarters_dict_list, sector_dict_list

def mongodb_update(data, mongodb_uri: str, database_name: str, collection_name: str):
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Insert data into MongoDB collection
    collection.insert_many(data)

    print('You have your data in Mongodb!!')
    
def save_to_json_file(data, filename):
    # File path where you want to save the JSON file
    file_path = f"{filename}.json"

    # Write the JSON list to the file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("JSON list saved to", file_path)

    return 1
# Pull data from API
data = pull_data()

# Structure the data
value_dict_list, earnings_dict_list, quarters_dict_list, sector_dict_list = structured_data(data)

# Provide the path to your JSON file, MongoDB URI, database name, and collection name
json_file_path = 'structureddata.json'
db_name = 'earnings'
collection_name = 'earnings_collection'

# Save structured data to JSON file
save_to_json_file(value_dict_list, "structureddata")

# Call the function to insert data into MongoDB
mongodb_update(value_dict_list, mongodb_uri, db_name, collection_name)

# Load structured data from JSON file into a DataFrame
earningsdf = pd.read_json("structureddata.json")


# Replace 'your_username' and 'your_password' with your PostgreSQL username and password
username = 'postgres'
password = '12345678'
host = 'localhost'
port = '5432'
database_name = 'postgres'
# Database connection URL
db_url = f'postgresql://{username}:{password}@{host}:{port}/{database_name}'
# Create a connection string to your PostgreSQL database
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database_name}')

# Store the DataFrame in PostgreSQL
earningsdf.to_sql('structureddata', engine, if_exists='replace', index=False)

# Confirming that the DataFrame has been stored in PostgreSQL by reading it back
earningsdf_from_sql = pd.read_sql_table('structureddata', engine)
print(earningsdf_from_sql.head())  # Display the first few rows of the DataFrame read from PostgreSQL


    

        
        
      
#Code for AveragePrice
            
# Function to fetch CSV data from URL and save it locally
def fetch_csv_from_url(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# URL of the CSV file
csv_url = 'https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/HSQ06/CSV/1.0/en'

# Name of the file to save locally
csv_file_path = 'AveragePrice.csv'

# Fetch CSV data from URL and save it locally
fetch_csv_from_url(csv_url, csv_file_path)


def csv_to_json(csv_file_path):
    data = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return json.dumps(data, indent=4)

csv_file_path ='AveragePrice.csv'
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



# Drop the specified columns
columns_to_drop = ["UNIT", "C02343V02817"]
df = df.drop(columns=columns_to_drop)

# Drop the '_id' and 'STATISTIC' columns
columns_to_drop = ['_id', 'ï»¿"STATISTIC"']
df = df.drop(columns=columns_to_drop)



# Convert the "VALUE" column to integer, handling errors by coercing invalid values to NaN
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce').astype('Int64')



# Convert 'Area' column to categorical
df['Area'] = df['Area'].astype('category')

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



print(df.info())
print(df.describe())



# Replace labels in the "Statistic Label" column
df['Statistic Label'] = df['Statistic Label'].replace({
    'New House Prices': 1,
    'Second Hand House Prices': 2
})

# Replace labels in the "Area" column
df['Area'] = df['Area'].replace({
    'Dublin': 1,
    'Cork': 2,
    'Galway': 3,
    'Limerick': 4,
    'Waterford': 5,
    'Other areas': 6,
    'National': 7
})


# Convert non-numeric data to numeric (if possible) for correlation computation

df['TLIST(Q1)'] = pd.to_numeric(df['TLIST(Q1)'], errors='coerce')
# This will convert non-numeric values to NaN, you can handle NaN values as per your requirement

#

# Convert 'Area' column to integers
df['Area'] = df['Area'].astype('category').cat.codes

# Print the updated DataFrame
print(df)

# Selecting only the columns with numerical data types
numerical_df = df[['VALUE', 'Statistic Label', 'Area']]



# Convert 'Area' column to integers
numerical_df['Area'] = df['Area'].astype('category').cat.codes

# Print the updated DataFrame
print(numerical_df)



# PostgreSQL credentials
#username = 'postgres'
#password = '12345678'
#host = 'localhost'  
#port = '5432' 
#database = 'averageprices'


# Replace 'your_username' and 'your_password' with your PostgreSQL username and password
username = 'postgres'
password = '12345678'

# Replace 'localhost' with your PostgreSQL host address if necessary
host = 'localhost'

# Replace '5432' with your PostgreSQL port if necessary
port = '5432'

# Database name
database_name = 'postgres'

# Database connection URL
db_url = f'postgresql://{username}:{password}@{host}:{port}/{database_name}'
# Create a connection string to your PostgreSQL database
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')

# Store the DataFrame in PostgreSQL
df.to_sql('averageprices', engine, if_exists='replace', index=False)

# Confirming that the DataFrame has been stored in PostgreSQL by reading it back
df_from_sql = pd.read_sql_table('averageprices', engine)
print(df_from_sql.head())  # Display the first few rows of the DataFrame read from PostgreSQL

#Code CA2

fname=os.path.join("FIM09.20240413155658.csv")
interestdf=pd.read_csv(fname, index_col=None)

data_dict=interestdf.to_dict("records")
mongoClient = MongoClient()
db = mongoClient['CA2']
collection = db['interest']


# Insert and update the collection with the information from the dataframe

try:
    for documento in data_dict:
        collection.update_one({'_id': documento['Month'] + str(documento['C02567V03112'])}, {"$set": documento}, upsert=True)
except:
    print("The insert / update function has not worked")

#Extract the data from MongoDB. Start by connecting again with MondgoDB

clientMongo=MongoClient()
database=clientMongo['CA2']
collectionMongo=database['interest']

interest_data=[]
for document in collectionMongo.find({}):
    interest_data.append(document)

interestdf=pd.DataFrame(interest_data)

#Remove the columns that are not of our interest
interestdf=interestdf.drop(columns=['_id','STATISTIC', 'Statistic Label', 'Interest Rate', 'Month', 'UNIT'])

#Rename the columns
interestdf=interestdf.rename(columns={'TLIST(M1)':'date','VALUE':'interest', 'C02567V03112':'interest_type'})

# Use pivot_table for creating new columns for the different values contained within the dafafame interestdf
df_grouped = interestdf.pivot_table(index='date', columns='interest_type', values='interest', aggfunc='first')

# Restore the format of the dataframe
df_grouped.reset_index(inplace=True)

# Rename the columns
df_grouped=df_grouped.rename(columns={1:'cbank_rediscount',3:'ebill_yield', 4:'buildings_mortgage'})

#Explore the data
df_grouped.shape

#Review if there are missing values
print(df_grouped.isnull().sum())

#Drop missing values
df_grouped=df_grouped.dropna(axis=0)

#Separate the date into month and year
df_grouped['year']=df_grouped['date'] /100
df_grouped['month']=df_grouped['date'] %100
df_grouped['year']=df_grouped['year'].astype(int)
df_grouped=df_grouped.drop(columns=['date'])
Qinterestdf = df_grouped[(df_grouped['month'] == 3) | (df_grouped['month'] == 6) | (df_grouped['month'] == 9)|
                         (df_grouped['month'] == 12)]

#Build the indicator of quarterly data that will be relevant for the visualization of the data

def month_to_quarter(mes):
    if mes in range(1, 4):
        return "Q1"
    elif mes in range(4, 7):
        return "Q2"
    elif mes in range(7, 10):
        return "Q3"
    else:
        return "Q4"

# Create a copy of the datafame
Qinterestdf = Qinterestdf.copy()

# Apply the function loc to transform the information contained within the field month to quarter
Qinterestdf.loc[:, 'quarter'] = Qinterestdf['month'].apply(month_to_quarter)

# Create a second copy of the datafame
Qinterestdf = Qinterestdf.copy()

# Modify the copy of the dataframe
Qinterestdf['quarter_year'] = Qinterestdf['year'].astype(str) + Qinterestdf['quarter'].astype(str)

# Reset index
print(Qinterestdf.columns.tolist())

# Replace 'your_username' and 'your_password' with your PostgreSQL username and password
username = 'postgres'
password = '12345678'

# Replace 'localhost' with your PostgreSQL host address if necessary
host = 'localhost'

# Replace '5432' with your PostgreSQL port if necessary
port = '5432'

# Database name
database_name = 'postgres'

# Store the DataFrame in PostgreSQL
Qinterestdf.to_sql('interests', engine, if_exists='replace', index=False)

# Confirming that the DataFrame has been stored in PostgreSQL by reading it back
df_from_sql = pd.read_sql_table('interests', engine)
print(df_from_sql.head())  # Display the first few rows of the DataFrame read from PostgreSQL

username = 'postgres'
password = '12345678'

# Replace 'localhost' with your PostgreSQL host address if necessary
host = 'localhost'

# Replace '5432' with your PostgreSQL port if necessary
port = '5432'

# Database name
database = 'postgres'

# Function to connect to PostgreSQL and verify table existence
def verify_table_existence(table_name):
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(user=username,
                                      password=password,
                                      host=host,
                                      port=port,
                                      database=database)

        # Create a cursor object
        cursor = connection.cursor()

        # SQL query to check if the table exists
        query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}')"

        # Execute the SQL query
        cursor.execute(query)

        # Fetch result
        table_exists = cursor.fetchone()[0]

        if table_exists:
            print(f"Table '{table_name}' exists in the database.")
        else:
            print(f"Table '{table_name}' does not exist in the database.")

    except Error as e:
        print("Error while connecting to PostgreSQL:", e)

    finally:
        # Close the cursor and connection
        if connection:
            cursor.close()
            connection.close()

# Verify existence of 'earnings' table
verify_table_existence('structureddata')

# Verify existence of 'averageprices' table
verify_table_existence('averageprices')

# Verify existence of 'interest' table
verify_table_existence('interests')

# Defining engine
engine = create_engine('postgresql://postgres:12345678@localhost:5432/postgres')

# SQL query to join the tables based on the common column "quarter"
sql_query = """
SELECT *
FROM averageprices
JOIN interests ON averageprices."Quarter" = interests.quarter_year
"""

# Execute the SQL query and load the result into a DataFrame
merged_df = pd.read_sql_query(sql_query, engine)

# Print the first few rows of the merged DataFrame
print(merged_df.head())


# Execute the SQL query and read the result into a DataFrame
merged_df = pd.read_sql_query(sql_query, engine)

# Display the first few rows of the merged DataFrame
print(merged_df.head())

# Export the merged DataFrame to a CSV file
merged_df.to_csv('merged_data.csv', index=False)