
#Import a CSV file and export it to Mongodb
import os
from pymongo import MongoClient
import pandas as pd
import psycopg2
import sqlalchemy
import plotly.graph_objects as go


fname=os.path.join(r"D:\Ordenador nuevo\Estudios\NCI\Database for Analytics Programming\CA2\FIM09.20240413155658.csv")
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

#Send the datafame to Postgresql and return an error if it has not correctly

try:
    connect=psycopg2.connect(dbname="postgres", user="postgres", password="061286",host="localhost",port="5432")

except:
    print("Connection with PostreSQL has failed")

engine= sqlalchemy.create_engine('postgresql://postgres:061286@localhost:5432/interests')
Qinterestdf.to_sql('interests', engine, if_exists='replace', index=False)

#Visualization
print(Qinterestdf.head(5))
#Create the scatter chart

# Create traces for every column
traces = []
for column in ['cbank_rediscount', 'ebill_yield', 'buildings_mortgage']:
    trace = go.Scatter(
        x=Qinterestdf['year'],
        y=Qinterestdf[column],
        mode='lines+markers',
        name=column
    )
    traces.append(trace)

# Create the layout of the graph
layout = go.Layout(
    title='Year trends of the variables',
    xaxis=dict(title='Año'),
    yaxis=dict(title='Valor')
)

# Create the object and add the traces and the layout
fig = go.Figure(data=traces, layout=layout)

# Print the graph
fig.show()


