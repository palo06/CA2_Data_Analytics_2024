@author: maria
@author: angeliki
@author: Paloma
"""

import requests
import csv
import json
import sqlalchemy
import numpy as np
from pymongo import MongoClient
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
import os
import psycopg2
from psycopg2 import Error
import plotly.graph_objects as go
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.stats.outliers_influence as oi
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor




# PostgreSQL universal credentials
username = 'postgres'
password = '12345678'
host = 'localhost'
port = '5432'
database = 'postgres'

# MondoDB universal credentials
mongodb_uri = 'mongodb://localhost:27017'
mongodb_name = ''

#Code for dataset "Earnings"

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


    

        
        
      
#Code for dataset "Average_Prices"
            
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
columns_to_drop = ["UNIT","TLIST(Q1)", "C02343V02817"]
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

#Code for dataset " Interests"

# Function to fetch CSV data from URL and save it locally
def fetch_csv_from_url(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# URL of the CSV file
csv_url = 'https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/FIM09/CSV/1.0/en'

# Name of the file to save locally
csv_file_path = 'FIM09.20240413155658.csv'

# Fetch CSV data from URL and save it locally
fetch_csv_from_url(csv_url, csv_file_path)


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

# SQL query to join the tables based on the common column "quarter" (Interests& Average_ Price)
sql_query = """
SELECT *
FROM averageprices
JOIN interests ON averageprices."Quarter" = interests.quarter_year
"""

# Execute the SQL query and load the result into a DataFrame
interests_price_merged_df = pd.read_sql_query(sql_query, engine)

# Drop the "Quarter" column from the DataFrame
interests_price_merged_df.drop(columns=['Quarter'], inplace=True)
# Print the first few rows of the merged DataFrame
print(interests_price_merged_df.head())


# Execute the SQL query and read the result into a DataFrame
interests_price_merged_df = pd.read_sql_query(sql_query, engine)




#Merging Average Prices & Earnings

engine = create_engine('postgresql://postgres:12345678@localhost:5432/postgres')

sql_query = """
SELECT *
FROM averageprices
JOIN structureddata ON averageprices."Quarter" = structureddata."quarter"
"""

try:
    merged_df = pd.read_sql_query(sql_query, engine)
    print(merged_df.head())
except Exception as e:
    print("Error exporting CSV file:", e)
    

# Drop the "Quarter" column from the DataFrame
merged_df.drop(columns=['Quarter'], inplace=True)

interests_price_merged_df.rename(columns={'VALUE': 'House_Price','Statistic Label': 'House_Type'}, inplace=True)
merged_df.rename(columns={'VALUE': 'House_Price', 'value': 'earnings_value','Statistic Label': 'House_Type'}, inplace=True)

    



# Visualizations for Interests_Price

# Histogram for numerical columns
numerical_columns = interests_price_merged_df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=interests_price_merged_df, x=column, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Count plot for categorical columns
categorical_columns = interests_price_merged_df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=interests_price_merged_df, x=column)
    plt.title(f'Count Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
# Select only numeric columns
numeric_columns = interests_price_merged_df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



#Linear Regression to predict House (Using the dataset Interests & House Price)
#Model 1
# Select predictor variables (X) and target variable (y)
X = interests_price_merged_df[['House_Type', 'Area', 'year', 'month','cbank_rediscount','ebill_yield','buildings_mortgage']]  # Predictor variables
y = interests_price_merged_df['House_Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#Multilinear Regression to predict House (Using the dataset Interests & House Price)


# Model 2
# Independent variables
independent_variables = ['House_Type', 'Area', 'year', 'month', 'cbank_rediscount', 'ebill_yield', 'buildings_mortgage']

# Add a constant term to the independent variables matrix (required for the intercept)
X = sm.add_constant(interests_price_merged_df[independent_variables])

# Dependent variable
y = interests_price_merged_df['House_Price']

# Fit the multilinear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Assumption 1: Checking linearity
for column in independent_variables:
    plt.figure(figsize=(8, 6))
    plt.scatter(interests_price_merged_df[column], interests_price_merged_df['House_Price'], alpha=0.5)
    plt.title(f'Scatterplot of House_Price vs {column}')
    plt.xlabel(column)
    plt.ylabel('House_Price')
    plt.grid(True)
    plt.show()

# Assumption 2: Checking multicollinearity
correlation_matrix = interests_price_merged_df[independent_variables].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# Assumption 3 & 4: Checking homoscedasticity and independence of residuals
residuals = model.resid
y_pred = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Durbin-Watson test for autocorrelation
durbin_watson_statistic = durbin_watson(residuals)
print("Durbin-Watson statistic:", durbin_watson_statistic)

# Assumption 5: Checking normality of residuals using Q-Q plot
sm.qqplot(residuals, line='s', color='skyblue')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Assumption 6: Checking influential cases using Cook's distance
influence = model.get_influence()
cooks_distance = influence.cooks_distance[0]

plt.figure(figsize=(8, 6))
plt.stem(cooks_distance, markerfmt=",", linefmt='b-', basefmt='r-')
plt.title("Cook's Distance Plot")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.show()

# Identify and remove influential cases
influential_cases = (cooks_distance > 4 / len(X))  # Threshold for influential cases
influential_cases_indices = X.index[influential_cases]
new_filtered_df = interests_price_merged_df.drop(influential_cases_indices)

# Print the shape of the filtered DataFrame after removing influential cases
print("Shape of filtered DataFrame:", new_filtered_df.shape)

# Model 3: Removing 'cbank_rediscount' and 'ebill_yield' variables
# Independent variables
independent_variables = ['House_Type', 'Area', 'year', 'month', 'buildings_mortgage']

# Add a constant term to the independent variables matrix (required for the intercept)
X = sm.add_constant(new_filtered_df[independent_variables])

# Dependent variable
y = new_filtered_df['House_Price']

# Fit the multilinear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Novelty to the prediction: Implementing Ensemble learninbg for house price prediction using Random Forest algorithm

# Select only numerical columns from interests_price_merged_df
numerical_columns = interests_price_merged_df.select_dtypes(include=[np.number])

X = numerical_columns.drop('House_Price', axis=1)  # Features
y = interests_price_merged_df['House_Price']  # Target variable

# Reset index to ensure alignment
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y to a DataFrame
y_train_df = pd.DataFrame(y_train, columns=['House_Price'])

# Define base models
base_models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    DecisionTreeRegressor(random_state=42)
]

# Train base models
predictions = []
for model in base_models:
    model.fit(X_train, y_train_df.values.ravel())  # Reshape y_train_df
    predictions.append(model.predict(X_test))

# Combine predictions using averaging
ensemble_predictions = np.mean(predictions, axis=0)

# Evaluate the ensemble model
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
print("Ensemble Mean Squared Error:", ensemble_mse)

# Visualizations for Earnings_Price
# Histogram for numerical columns
numerical_columns = merged_df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=merged_df, x=column, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Count plot for categorical columns
categorical_columns = merged_df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=merged_df, x=column)
    plt.title(f'Count Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
# Select only numeric columns
numeric_columns = merged_df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#Linear Regression to predict House (Using the dataset Earnings & House Price)
#Model 1
# Select predictor variables (X) and target variable (y)
X = merged_df[['House_Type', 'Area','earnings_value']]  # Predictor variables
y = merged_df['House_Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#Multilinear Regression to predict House (Using the dataset Earnings & House Price)


# Model 2
# Independent variables
independent_variables = ['House_Type', 'Area','earnings_value']

# Add a constant term to the independent variables matrix (required for the intercept)
X = sm.add_constant(merged_df[independent_variables])

# Dependent variable
y = merged_df['House_Price']

# Fit the multilinear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())

# Assumption 1: Checking linearity
for column in independent_variables:
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_df[column], merged_df['House_Price'], alpha=0.5)
    plt.title(f'Scatterplot of House_Price vs {column}')
    plt.xlabel(column)
    plt.ylabel('House_Price')
    plt.grid(True)
    plt.show()

# Assumption 2: Checking multicollinearity
correlation_matrix = merged_df[independent_variables].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Independent Variables')
plt.show()

# Assumption 3 & 4: Checking homoscedasticity and independence of residuals
residuals = model.resid
y_pred = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Durbin-Watson test for autocorrelation
durbin_watson_statistic = durbin_watson(residuals)
print("Durbin-Watson statistic:", durbin_watson_statistic)

# Assumption 5: Checking normality of residuals using Q-Q plot
sm.qqplot(residuals, line='s', color='skyblue')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

# Assumption 6: Checking influential cases using Cook's distance
influence = model.get_influence()
cooks_distance = influence.cooks_distance[0]

plt.figure(figsize=(8, 6))
plt.stem(cooks_distance, markerfmt=",", linefmt='b-', basefmt='r-')
plt.title("Cook's Distance Plot")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.show()

# Identify and remove influential cases
influential_cases = (cooks_distance > 4 / len(X))  # Threshold for influential cases
influential_cases_indices = X.index[influential_cases]
new_earnings_filtered_df = merged_df.drop(influential_cases_indices)

# Print the shape of the filtered DataFrame after removing influential cases
print("Shape of filtered DataFrame:", new_earnings_filtered_df.shape)

# Model 3: Using the daframe without the influential cases
# Independent variables
independent_variables = ['House_Type', 'Area','earnings_value']

# Add a constant term to the independent variables matrix (required for the intercept)
X = sm.add_constant(new_earnings_filtered_df[independent_variables])

# Dependent variable
y = new_earnings_filtered_df['House_Price']

# Fit the multilinear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the model
print(model.summary())





# Export the merged DataFrames to a CSV file
interests_price_merged_df.to_csv('merged_data_ interests_house_price.csv', index=False) #--> Interests & House_Price
merged_df.to_csv('merged_data_earnings_house_price.csv', index=False) #---> Earnings & House_Price
