#build simple app that takes data from user, takes city/sensor location (from preset list), and displays aqi data

# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, url_for, redirect, jsonify
import requests
from enum import unique
import sqlalchemy
from typing import Iterator, Dict, Any
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, select, MetaData, Table
from sqlalchemy.orm import sessionmaker
import requests
import config
import datetime
import json
from json import JSONEncoder
import os
from calendar import EPOCH
import asyncio
import json
import requests
import requests
from requests.structures import CaseInsensitiveDict
import traceback
import logging
from datetime import date
from sqlalchemy.ext.declarative import declarative_base
engine = sqlalchemy.create_engine(config.DATABASE_URI)
Session = sessionmaker(bind=engine)
app = Flask(__name__)
import time
 
    

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/get_sensor_cities', methods=['GET', 'POST'])
def get_db_sensor_cities():
    # POST request
    if request.method == 'POST':
        print('Incoming..')
        print(request.get_json())  # parse as JSON
        results = {'processed': 'true'}
        return results

    # GET request
    else:
        sensor_cities = get_sensor_cities()
        return sensor_cities
    

@app.route('/get_aqi_data', methods = ['POST'])
def get_user_selected_variables():
    jsdata = request.form

    city = jsdata.getlist('city')[0]
    start_date = jsdata.getlist('start_date')[0]
    end_date = jsdata.getlist('end_date')[0]
    
    historical_data = asyncio.run(query_db_with_user_inputs(city, start_date, end_date))
    return historical_data


def create_aqi_history_table(cursor) -> None:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aqi_history_table (
            sensor_city             TEXT,
            date                   DATE,
            pm25                    INT,
            pm10                    INT,
            o3                      INT,
            no2                     INT,
            so2                     INT,
            co                      INT,
            aqi                     INT,
            aqi_classification      TEXT,
            UNIQUE(sensor_city, date)
            );
     """)

def aqi_history_insert_execute_batch_iterator(connection, json_data: Iterator[Dict[str, Any]]) -> None:
    with config.CONNECTION.cursor() as cursor:
        create_aqi_history_table(cursor)
        iter_json = ({
            **vals,
            "sensor_city": vals["sensor_city"],
            "date": vals["date"],
            "pm25": vals["pm25"],
            "pm10": vals["pm10"],
            "o3": vals["o3"],
            "no2": vals["no2"],
            "so2": vals["so2"],
            "co": vals["co"],
            "aqi": vals["aqi"],
            "aqi_classification": vals["aqi_classification"]
        } for vals in json_data)
        
        psycopg2.extras.execute_batch(cursor, """
            INSERT INTO aqi_history_table VALUES (
                %(sensor_city)s,
                %(date)s,
                %(pm25)s,
                %(pm10)s,
                %(o3)s,
                %(no2)s,
                %(so2)s,
                %(co)s,
                %(aqi)s,
                %(aqi_classification)s
            )
            ON CONFLICT (sensor_city, date) DO NOTHING; """, iter_json)
    
    
def add_to_aqi_table(df):
    config.CONNECTION.autocommit = True
    # with config.CONNECTION.cursor() as cursor:
    #     cursor.execute("""
    #         DROP TABLE IF EXISTS aqi_history_table """)
        
    aqi_history_insert_execute_batch_iterator(config.CONNECTION, df)
        
def get_sensor_cities():
        
    new_connection = psycopg2.connect(
            host= config.HOST,
            database= config.DATABASE,
            user= config.USER,
            password= config.PASSWORD
        )

    #config.CONNECTION.autocommit = True
    new_cursor = new_connection.cursor()
    query = """SELECT DISTINCT sensor_city FROM aqi_history_table """                
    new_cursor.execute(query)
    new_connection.commit()
    names = [ x[0] for x in new_cursor.description]
    rows = new_cursor.fetchall()
    resulting_df = pd.DataFrame(rows, columns=names)
    sensor_city_json = resulting_df.to_json()
    print(sensor_city_json)
    new_cursor.close()
    new_connection.close()
    
    return sensor_city_json


def query_existing_table(sensor_city, start_date, end_date):
    
    if len(sensor_city)==0:
        return 0, []
    if len(end_date) == 0:
        end_date = start_date
        
    print("getting historical data...")
    new_connection = config.CONNECTION

    #config.CONNECTION.autocommit = True
    new_cursor = new_connection.cursor()
    get_data_by_city_only = """SELECT * FROM aqi_history_table WHERE sensor_city = '""" + sensor_city + """' """                
    get_data_by_all_variables = """SELECT * FROM public.aqi_history_table WHERE sensor_city =  '""" + sensor_city + """' """ + """ AND date >= '""" + start_date + """' """ + """ AND date <= '""" + end_date + """' """

    print("query: ", get_data_by_all_variables)
    new_cursor.execute(get_data_by_all_variables)
    new_connection.commit()
    names = [ x[0] for x in new_cursor.description]
    rows = new_cursor.fetchall()
    resulting_df = pd.DataFrame( rows, columns=names)
    print("query results \n: ", resulting_df)
    new_cursor.close()
    new_connection.close()
    return resulting_df
    
 

def equation_one_to_caculate_index(I_hi,I_lo,BP_hi,BP_lo,Cp):
    
    if BP_hi-BP_lo == 0: #division by zero:
        return 0
    else:
        Ip = (I_hi-I_lo)/(BP_hi-BP_lo) * (Cp-BP_lo) + I_lo
        return int(Ip)
    
       
def calculating_aqi(particulate_type, concentration_number, reporting_interval):
    
    I_hi = 0 #AQI value corresponding to BPHi
    I_lo = 0 #AQI value corresponding to BPLo
    BP_hi = 0 #concentration breakpoint that is greater than or equal to Cp
    BP_lo = 0 #concentration breakpoint that is less than or equal to Cp
    Cp = 0  #truncated concentration of pollutant p
    
    
    if particulate_type == "ozone":
        Cp = round(concentration_number,3)
    elif particulate_type == "pm25":
        Cp = round(concentration_number,1)
    elif particulate_type == "pm10":
        Cp = int(concentration_number)
    elif particulate_type == "co":
        Cp = round(concentration_number,1)
    elif particulate_type == "so2":
        Cp = int(concentration_number)
    elif particulate_type == "no2":
        Cp = int(concentration_number)
        
    
    for breakpoint in aqi_table[particulate_type][reporting_interval]:
        if Cp >= breakpoint["low"] and Cp <= breakpoint["high"]:
            I_hi = breakpoint["aqi_low"] #AQI value corresponding to BPHi
            I_lo =breakpoint["aqi_high"] #AQI value corresponding to BPLo
            BP_hi = breakpoint["low"] #concentration breakpoint that is greater than or equal to Cp
            BP_lo = breakpoint["high"] #concentration breakpoint that is less than or equal to Cp
    
    aqi = equation_one_to_caculate_index(I_hi,I_lo,BP_hi,BP_lo,Cp)
    return aqi
    
        
        
    """rules to adhere to:
    
    Step 1: Identify the highest concentration among all of the monitors within each reporting area and truncate
as follows:
    
        Ozone (ppm) – truncate to 3 decimal places
        PM2.5 (µg/m3) – truncate to 1 decimal place
        PM10 (µg/m3) – truncate to integer
        CO (ppm) – truncate to 1 decimal place
        SO2 (ppb) – truncate to integer
        NO2 (ppb) – truncate to integer
        
        note:  if you have both O3_ppm_8_hr and O3_ppm_1_hr, calculate both and take the max
    
    Step 2: use the aqi_table to find breakpoints containing that concentration
    
    Step 3: using equation 1, calculate the index
    
    Step 4: round the index to the nearest integer


   
    
    """
    
    
def aqi_category_mapping(df):
    
    conditions = [
        (df['aqi'] <= 50),
        (df['aqi'] >50) & (df['aqi']<=100),
        (df['aqi'] >100) & (df['aqi']<=150),
        (df['aqi'] >150) & (df['aqi']<=200),
        (df['aqi'] >200) & (df['aqi']<=300),
        (df['aqi'] >300) & (df['aqi']<=500),
        (df['aqi'] <= 500)
    ]
    values = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very unhealthy", "Hazardous", "Not possible"]
    df["aqi_classification"] = np.select(conditions, values)
    df.head()
    print(df)
    return df

def create_no2_table(aqi_table):
    aqi_table["no2"] = [
        {"1-hour": [{
            "low": 0,
            "high": 53,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 54,
            "high": 100,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
             "low": 101,
            "high": 360,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 361,
            "high": 649,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 650,
            "high": 1249,
            "aqi_low": 201,
            "aqi_high": 300
        }, 
        {
            "low": 1250,
            "high": 1649,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 1650,
            "high": 2049,
            "aqi_low": 401,
            "aqi_high": 500
        }]      
    }]
    return aqi_table

def create_so2_table(aqi_table):
      
    aqi_table["so2"] = [{
        "1-hour": [{
            "low": 0,
            "high": 35,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 36,
            "high": 75,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
            "low": 76,
            "high": 185,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 186,
            "high": 304,
            "aqi_low": 151,
            "aqi_high": 200
        }],
        "24-hour": [{
            "low": 305,
            "high": 604,
            "aqi_low": 201,
            "aqi_high": 300
        }, 
        {
            "low": 605,
            "high": 504,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 505,
            "high": 604,
            "aqi_low": 401,
            "aqi_high": 500
        }]
    }] 
    
    return aqi_table

def create_co_table(aqi_table):
    aqi_table["co"] = [
        {"8-hour": [{
            "low": 0.0,
            "high": 4.4,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 4.5,
            "high": 9.4,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
            "low": 9.5,
            "high": 12.4,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 12.5,
            "high": 15.4,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 15.5,
            "high": 30.4,
            "aqi_low": 201,
            "aqi_high": 300
        }, 
        {
            "low": 30.5,
            "high": 40.4,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 40.5,
            "high": 50.4,
            "aqi_low": 401,
            "aqi_high": 500
        }]      
    }]
    return aqi_table
    
def create_pm10_table(aqi_table):
        
    aqi_table["pm10"] = [
        {"24-hour": [{
            "low": 0,
            "high": 54,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 55,
            "high": 154,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
             "low": 155,
            "high": 254,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 255,
            "high": 354,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 355,
            "high": 424,
            "aqi_low": 201,
            "aqi_high": 300
        }, 
        {
            "low": 425,
            "high": 504,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 505,
            "high": 604,
            "aqi_low": 401,
            "aqi_high": 500
        }]      
    }]
    return aqi_table

def create_pm25_table(aqi_table):
    
    aqi_table["pm25"] = [
        {"24-hour": [{
            "low": 0.0,
            "high": 12.0,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 12.1,
            "high": 35.4,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
            "low": 35.5,
            "high": 55.4,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 55.5,
            "high": 150.4,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 150.5,
            "high": 250.4,
            "aqi_low": 201,
            "aqi_high": 300
        }, 
        {
            "low": 250.5,
            "high": 350.4,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 350.5,
            "high": 500.4,
            "aqi_low": 401,
            "aqi_high": 500
        }]      
    }]
    return aqi_table
   
def create_ozone_table(aqi_table):
    aqi_table["ozone"] = {}
    
    aqi_table["ozone"]["8-hour"] = [{
            "low": 0.000,
            "high": 0.054,
            "aqi_low": 0,
            "aqi_high": 50
        },
        {
            "low": 0.055,
            "high": 0.070,
            "aqi_low": 51,
            "aqi_high": 100
        },
        {
            "low": 0.071,
            "high": 0.085,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 0.086,
            "high": 0.105,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 0.106,
            "high": 0.200,
            "aqi_low": 201,
            "aqi_high": 300
        }  
        ]
    aqi_table["ozone"]["1-hour"] = [{
            "low": 0.125,
            "high": 0.164,
            "aqi_low": 101,
            "aqi_high": 150
        },
        {
            "low": 0.165,
            "high": 0.204,
            "aqi_low": 151,
            "aqi_high": 200
        },
        {
            "low": 0.205,
            "high": 0.404,
            "aqi_low": 201,
            "aqi_high": 300
        },
        {
            "low": 0.405,
            "high": 0.504,
            "aqi_low": 301,
            "aqi_high": 400
        },
        {
            "low": 0.505,
            "high": 0.604,
            "aqi_low": 401,
            "aqi_high": 500
        }  
        ]
    
    return aqi_table
    
     
def create_aqi_table():
    #see page 14 here https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    #according to epa guidance https://www.epa.gov/outdoor-air-quality-data/how-aqi-calculated
    
    global aqi_table
    aqi_table = {}
    aqi_table = create_ozone_table(aqi_table)
    aqi_table = create_pm25_table(aqi_table)
    aqi_table = create_pm10_table(aqi_table)
    aqi_table = create_co_table(aqi_table)
    aqi_table = create_so2_table(aqi_table)
    aqi_table = create_no2_table(aqi_table)
    
    return aqi_table
    

def alchemyencoder(obj):
    if isinstance(obj, datetime.date):
        return obj.isoformat()


    
async def query_db_with_user_inputs(sensor_city, start_date, end_date):

   
    if len(sensor_city)==0:
        return 0, []
    if len(end_date) == 0:
        end_date = start_date
        
    print("getting historical data...")
    new_connection = psycopg2.connect(
            host= config.HOST,
            database= config.DATABASE,
            user= config.USER,
            password= config.PASSWORD
        )
    

    new_cursor = new_connection.cursor()
    get_data_by_city_only = """SELECT * FROM aqi_history_table WHERE sensor_city = '""" + sensor_city + """' """                
    get_data_by_all_variables = """SELECT * FROM public.aqi_history_table WHERE sensor_city =  '""" + sensor_city + """' """ + """ AND date >= '""" + start_date + """' """ + """ AND date <= '""" + end_date + """' """
    new_cursor.execute(get_data_by_all_variables)
    new_connection.commit()
    
    result = engine.execute(get_data_by_all_variables)
    json_data = json.dumps([dict(r) for r in result], default=alchemyencoder)
    print(" json data: \n", json_data)

    new_cursor.close()
    new_connection.close()
    return json_data
    
    
    
def add_sample_data(df, sensor_city):
    
    all_particulate_types = ["pm25", "pm10", "o3", "no2", "so2", "co"]
    copy_df = df
    copy_df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')
    copy_df["date"] = df["date"].dt.date
    df.columns = df.columns.str.replace(' ', '')
    
    only_numeric_col_names = list(df.columns.values)
    only_numeric_col_names.remove('date')
    columns_to_add_later = list(set(all_particulate_types) - set(only_numeric_col_names))  #in order to have a full dataset

        
    only_numeric_columns = df[only_numeric_col_names]
        
    df = df[only_numeric_col_names].apply(lambda x: x.str.strip()).replace('',  None) #np.nan)    
    
    df["aqi"] = only_numeric_columns.max(axis=1).astype(int) #aqi is calculated as max(pm25, pm10, o3, etc) so take max aqi for each particulate for each day
    copy_df["date"] = copy_df["date"].astype(str)
    df.insert(0, "date", copy_df["date"])
    df.insert(1, "sensor_city", sensor_city)
    if len(columns_to_add_later) > 0:
        for particulate_type in columns_to_add_later:
            index_to_insert = all_particulate_types.index(particulate_type) + 2
            df.insert(index_to_insert, particulate_type, None)        
    
    df = aqi_category_mapping(df) 
    json_data = df.to_dict('records')
    # [{'date': '2022-12-02', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '31', 'pm10': '14', 'o3': '27', 'no2': nan, 'so2': nan, 'aqi': 31}, {'date': '2022-12-03', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '22', 'pm10': '21', 'o3': '12', 'no2': nan, 'so2': nan, 'aqi': 22}, {'date': '2022-12-04', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '48', 'pm10': '15', 'o3': '19', 'no2': nan, 'so2': nan, 'aqi': 48}, {'date': '2022-12-05', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '40', 'pm10': '16', 'o3': '28', 'no2': nan, 'so2': nan, 'aqi': 40}, {'date': '2022-12-06', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '33', 'pm10': '14', 'o3': '28', 'no2': nan, 'so2': nan, 'aqi': 33}, {'date': '2022-12-07', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '29', 'pm10': '13', 'o3': '29', 'no2': nan, 'so2': nan, 'aqi': 29}, {'date': '2022-12-08', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '27', 'pm10': '15', 'o3': '28', 'no2': nan, 'so2': nan, 'aqi': 28}, {'date': '2022-12-09', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '30', 'pm10': '18', 'o3': '26', 'no2': nan, 'so2': nan, 'aqi': 30}, {'date': '2022-12-10', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '38', 'pm10': '18', 'o3': '25', 'no2': nan, 'so2': nan, 'aqi': 38}, {'date': '2022-12-11', 'sensor_city': 'san-diego - sherman elementary school', 'pm25': '42', 'pm10': '16', 'o3': '25', 'no2': nan, 'so2': nan, 'aqi': 42}]
    # example of what data should look like before inserting into table
    add_to_aqi_table(json_data)
        
def delete_db():
    with config.CONNECTION as conn:
        with conn.cursor() as cursor:
            sql_drop_table_if_exists = """ DROP TABLE IF EXISTS aqi_history_table; """
            cursor.execute(sql_drop_table_if_exists)
            cursor.close()
        
        
async def add_test_data():    
    
    delete_db() #if exists
    files_and_sensor_cities = [
                {"file_name": "aqi_csv_data/san-diego - sherman elementary school, san diego, california-air-quality.csv", "sensor_city": "san-diego - sherman elementary school"},
                {"file_name": "aqi_csv_data/london-air-quality.csv", "sensor_city":  "london"},
                {"file_name": "aqi_csv_data/los-angeles-north main street-air-quality.csv", "sensor_city":  "los-angeles-north main street"},
                {"file_name": "aqi_csv_data/lourdes-lapacca, france-air-quality.csv", "sensor_city":  "lourdes-lapacca, france"},
                {"file_name": "aqi_csv_data/milano-senato, lombardia, italy-air-quality.csv", "sensor_city":  "milano-senato, lombardia, italy"},
                {"file_name": "aqi_csv_data/olive-st, seattle, washington, usa-air-quality.csv", "sensor_city": "olive-st, seattle, washington"},
                {"file_name": "aqi_csv_data/paris-air-quality.csv", "sensor_city":"paris"}
            ]
    for row in files_and_sensor_cities:
        #print("adding data from \n" + row + "\nto the table")
        file_name = row["file_name"]
        sensor_city = row["sensor_city"]
        df = pd.read_csv(file_name, index_col=False)
        add_sample_data(df, sensor_city)
    
    
    print("testing that db was populated with the following query\n")
    sensor_city = "san-diego - sherman elementary school"
    start_date = '2022-12-04'
    end_date = '2022-12-08'
    query_existing_table(sensor_city, start_date, end_date)
 

     
async def main():
    
    app.run(port=8000, debug=True)
    
    #if it's youre first run, run await add_test_data()
    #in all other instances, run app.run(port=8000, debug=True)
    
    

    

asyncio.run(main())
