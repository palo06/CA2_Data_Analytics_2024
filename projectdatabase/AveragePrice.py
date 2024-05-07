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
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import plotly.graph_objs as go

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



# Drop the specified columns
columns_to_drop = ["UNIT", "Quarter", "C02343V02817"]
df = df.drop(columns=columns_to_drop)

# Drop the '_id' and 'STATISTIC' columns
columns_to_drop = ['_id', 'ï»¿"STATISTIC"']
df = df.drop(columns=columns_to_drop)



# Convert the "VALUE" column to integer, handling errors by coercing invalid values to NaN
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce').astype('Int64')

# Convert 'House_Type' column to categorical
df['Statistic Label'] = df['Statistic Label'].astype('category')

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

#Renaming variables
df=df.rename(columns={"Statistic Label":"House_Type"})
print(df.info())
print(df.describe())



# Replace labels in the "Statistic Label" column
df['House_Type'] = df['House_Type'].replace({
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



# Convert 'House_Type' column to integers
df['House_Type'] = df['House_Type'].astype('category').cat.codes

# Convert 'Area' column to integers
df['Area'] = df['Area'].astype('category').cat.codes

# Print the updated DataFrame
print(df)

# Selecting only the columns with numerical data types
numerical_df = df[['VALUE', 'House_Type', 'Area']]

# Convert 'House_Type' column to integers
numerical_df['House_Type'] = df['House_Type'].astype('category').cat.codes

# Convert 'Area' column to integers
numerical_df['Area'] = df['Area'].astype('category').cat.codes

# Print the updated DataFrame
print(numerical_df)



# PostgreSQL credentials
username = 'postgres'
password = '12345678'
host = 'localhost'  
port = '5432' 
database = 'postgres'

# Create a connection string to your PostgreSQL database
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')

# Store the DataFrame in PostgreSQL
df.to_sql('average_prices', engine, if_exists='replace', index=False)

# Confirming that the DataFrame has been stored in PostgreSQL by reading it back
df_from_sql = pd.read_sql_table('average_prices', engine)
print(df_from_sql.head())  # Display the first few rows of the DataFrame read from PostgreSQL


# Plot histogram for 'VALUE'
plt.figure(figsize=(8, 6))
plt.hist(df_from_sql['VALUE'], bins=20, color='blue', alpha=0.7)
plt.xlabel('VALUE')
plt.ylabel('Frequency')
plt.title('Histogram of VALUE')
plt.grid(True)
plt.show()



# Plot histogram for 'House Type'
plt.figure(figsize=(8, 6))
plt.hist(df_from_sql['House_Type'], bins=20, color='purple', alpha=0.7)
plt.xlabel('House_Type')
plt.ylabel('Frequency')
plt.title('Histogram of House_Type')
plt.grid(True)
plt.show()

# Plot histogram for 'Area'
plt.figure(figsize=(8, 6))
plt.hist(df_from_sql['Area'], bins=20, color='orange', alpha=0.7)
plt.xlabel('Area')
plt.ylabel('Area')
plt.title('Histogram of Area')
plt.grid(True)
plt.show()

# Calculating the correlation matrix
correlation_matrix = df_from_sql.corr()

print(correlation_matrix)

print(df_from_sql.columns)

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
