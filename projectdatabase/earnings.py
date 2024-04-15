import requests
import json
import pandas as pd
from pymongo import MongoClient
import psycopg2
from psycopg2 import Error


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


def create_postgre_connection():

    # Connect to your PostgreSQL database
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="blackbaby",
            host="localhost",
            port="5432",
            database="postgres"
        )
        # cursor = connection.cursor()
        return connection

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
        return 0


if __name__ == '__main__':
    data = pull_data()
    value_dictionary_list, earnings_category_dict_list, quarters_dict_list, sector_dict_list = structured_data(data)

    save_to_json_file(value_dictionary_list, "values")
    save_to_json_file(earnings_category_dict_list, "earnings_category")
    save_to_json_file(quarters_dict_list, "quarters")
    save_to_json_file(sector_dict_list, "sectors")
    postgre_connection = create_postgre_connection()

    # print(pd.DataFrame(result))

    # Connection information to MongoDB
    mongodb_uri = "mongodb://localhost:27017/"
    database_name = "projectDatabase"

    # mongodb_update(value_dictionary_list, mongodb_uri, database_name, "projectdata")
    # mongodb_update(earnings_category_dict_list, mongodb_uri, database_name, "earningsdata")
    # mongodb_update(quarters_dict_list, mongodb_uri, database_name, "quartersdata")
    # mongodb_update(sector_dict_list, mongodb_uri, database_name, "sectordata")
