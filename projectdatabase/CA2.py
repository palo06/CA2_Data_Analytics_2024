
#Import a CSV file and export it to Mongodb
import os
from pymongo import MongoClient
import pandas as pd
import luigi


fname=os.path.join(r"D:\Ordenador nuevo\Estudios\NCI\Database for Analytics Programming\CA2\FIM09.20240413155658.csv")
interestdf=pd.read_csv(fname)
interestdf.reset_index(inplace=True)
data_dict=interestdf.to_dict("records")
mongoClient = MongoClient()
db = mongoClient['CA2']
collection = db['interest']
collection.insert_many(data_dict)

#Extract the data from MongoDB. Start by connecting again with MondgoDB

clientMongo=MongoClient()
database=clientMongo['CA2']
collectionMongo=database['interest']

interest_data=[]
for document in collectionMongo.find({}):
    interest_data.append(document)

interestdf=pd.DataFrame(interest_data)

#Review the columns
print(interestdf.head(10))

#Remove the columns that are not of our interest
interestdf=interestdf.drop(columns=['_id', 'index', 'STATISTIC', 'Statistic Label', 'C02567V03112', 'Interest Rate',
                                    'Month', 'UNIT'])
interestdf=interestdf.rename(columns={'TLIST(M1)':'date','VALUE':'interest'})
print(interestdf.head(10))
#interestdf.info()

#Explore the data
interestdf.shape

#Review if there are missing values
print(interestdf.isnull().sum())

#Pending to determine how to address the missing values
interestdf=interestdf.dropna(axis=0)


interestdf['year']= interestdf['date'] /100
interestdf['month']= interestdf['date'] %100
interestdf['year']=interestdf['year'].astype(int)
interestdf=interestdf.drop(columns=['date'])
print(interestdf.head(20))

#Pending to determine how to address the missing values

#interestdf.drop(columns=['month'==1,axis=1,inplace=True])
Qinterestdf = interestdf[(interestdf['month'] == 3) | (interestdf['month'] == 6) | (interestdf['month'] == 9)|
                         (interestdf['month'] == 12)]

print(Qinterestdf.head(20))