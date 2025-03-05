# Importing necessary libraries

from dagster import asset, Output, MetadataValue, multi_asset, AssetOut
from collections.abc import MutableMapping
import requests
import json
import pymongo
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
import plotly.graph_objects as go
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
import dash
from dash import html, dcc
from jupyter_dash import JupyterDash

# API endpoint for all datasets
url_cardiovascular = "https://data.cdc.gov/resource/rsk5-566a.json"  
url_tobacco = "https://data.cdc.gov/resource/wsas-xwh5.json"  
url_obesity = "https://data.cdc.gov/resource/hn4x-zwk7.json"

# Set limit and offset
limit = 1000
records = 150000
offset = range(0, records, limit)

# Asset to fetch cardiovascular disease death rates data
@asset
def fetch_cardiovascular_data():
    data = []
    try:
        for i in offset:
            find_url = f"{url_cardiovascular}?$limit={limit}&$offset={i}"
            response = requests.get(find_url)
            response.raise_for_status()
            result = response.json()
            data.extend(result)
        with open("cardiovascular.json", "w", encoding='utf-8') as jf:
            json.dump(data, jf, indent=3)
        return Output(value=data, metadata={"cardiovascular_data": MetadataValue.path("cardiovascular.json")})
    except requests.exceptions.HTTPError as httperror:
        raise Exception(f"HTTP error occurred: {httperror}")
    except requests.exceptions.RequestException as reqexc:
        raise Exception(f"Error occurred during request: {reqexc}")
    except Exception as e:
        raise Exception(f"Unknown error occurred: {e}")

# Asset to fetch tobacco rates data
@asset
def fetch_tobacco_data():
    data = []
    try:
        for i in offset:
            find_url = f"{url_tobacco}?$limit={limit}&$offset={i}"
            response = requests.get(find_url)
            response.raise_for_status()
            result = response.json()
            data.extend(result)
        with open("tobacco_use.json", "w", encoding='utf-8') as jf:
            json.dump(data, jf, indent=3)
        return Output(value=data, metadata={"tobacco_data": MetadataValue.path("tobacco_use.json")})
    except requests.exceptions.HTTPError as httperror:
        raise Exception(f"HTTP error occurred: {httperror}")
    except requests.exceptions.RequestException as reqexc:
        raise Exception(f"Error occurred during request: {reqexc}")
    except Exception as e:
        raise Exception(f"Unknown error occurred: {e}")

# Asset to fetch obesity data
@asset
def fetch_obesity_data():
    data = []
    try:
        for i in offset:
            find_url = f"{url_obesity}?$limit={limit}&$offset={i}"
            response = requests.get(find_url)
            response.raise_for_status()
            result = response.json()
            data.extend(result)
        with open("obesity.json", "w", encoding='utf-8') as jf:
            json.dump(data, jf, indent=3)
        return Output(value=data, metadata={"obesity_data": MetadataValue.path("obesity.json")})
    except requests.exceptions.HTTPError as httperror:
        raise Exception(f"HTTP error occurred: {httperror}")
    except requests.exceptions.RequestException as reqexc:
        raise Exception(f"Error occurred during request: {reqexc}")
    except Exception as e:
        raise Exception(f"Unknown error occurred: {e}")

# Asset to upload cardiovascular data to MongoDB
@asset
def upload_to_mongodb_cardiovascular(fetch_cardiovascular_data):
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['APDV_project_group9']
        collection_cardio = db['cardiovascular_disease']        
        batch_size = 10000
        for i in range(0, len(fetch_cardiovascular_data), batch_size):
            batch = fetch_cardiovascular_data[i:i+batch_size]
            collection_cardio.insert_many(batch)
        return Output(value="Cardiovascular data uploaded to MongoDB successfully.", metadata={"collection": "cardiovascular_disease"})
    except pymongo.errors.PyMongoError as pye:
        raise Exception(f"Could not connect to MongoDB due to the error: {pye}")
    except Exception as e:
        raise Exception(f"An error occurred during upload: {e}")

# Asset to upload tobacco data to MongoDB
@asset
def upload_to_mongodb_tobacco(fetch_tobacco_data):
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['APDV_project_group9']
        collection_tobacco = db['tobacco_use']
        batch_size = 10000
        for i in range(0, len(fetch_tobacco_data), batch_size):
            batch = fetch_tobacco_data[i:i+batch_size]
            collection_tobacco.insert_many(batch)
        return Output(value="Tobacco data uploaded to MongoDB successfully.", metadata={"collection": "tobacco_use"})
    except pymongo.errors.PyMongoError as pye:
        raise Exception(f"Could not connect to MongoDB due to the error: {pye}")
    except Exception as e:
        raise Exception(f"An error occurred during upload: {e}")

# Asset to upload obesity data to MongoDB
@asset
def upload_to_mongodb_obesity(fetch_obesity_data):
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['APDV_project_group9']
        collection_obesity = db['obesity']
        batch_size = 10000
        for i in range(0, len(fetch_obesity_data), batch_size):
            batch = fetch_obesity_data[i:i+batch_size]
            collection_obesity.insert_many(batch)
        return Output(value="Obesity data uploaded to MongoDB successfully.", metadata={"collection": "obesity"})
    except pymongo.errors.PyMongoError as pye:
        raise Exception(f"Could not connect to MongoDB due to the error: {pye}")
    except Exception as e:
        raise Exception(f"An error occurred during upload: {e}")

# Asset to load data from MongoDB
@asset
def load_data_from_mongodb():
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['APDV_project_group9']
        
        collection_cardio = db['cardiovascular_disease']
        collection_tobacco = db['tobacco_use']
        collection_obesity = db['obesity']
        
        cardio_df = pd.DataFrame(list(collection_cardio.find()))
        tobacco_df = pd.DataFrame(list(collection_tobacco.find()))
        obesity_df = pd.DataFrame(list(collection_obesity.find()))
        
        return {
            "cardio_df": cardio_df,
            "tobacco_df": tobacco_df,
            "obesity_df": obesity_df
        } 
    except Exception as e:
        raise Exception(f"An error occurred during data retrieval: {e}")

# Asset for EDA of cardiovascular data
@asset
def eda_cardio_data(load_data_from_mongodb):

    cardio_df = load_data_from_mongodb["cardio_df"]
    
    cardio_df.drop(columns=['_id'], inplace=True)  # Dropping id column assigned by MongoDB
    
    print(cardio_df.info(), "\n")
    print(cardio_df.shape, "\n")
    print(cardio_df.isnull().sum(), "\n")  # Checking for null values

    # Checking for ranged years
    cardio_df['year_new'] = cardio_df['year'].apply(lambda x: 'Single year' if '-' not in x else 'Ranged year')
    count = cardio_df['year_new'].value_counts()
    plt.figure(figsize=(12,8))
    count.plot(kind='barh', color=['cyan', 'yellow'], title='Single year vs Ranged years')
    plt.xlabel('Count')
    plt.ylabel('Year type')
    plt.show()

    cardio_df1 = cardio_df[['year', 'locationabbr', 'data_value', 'stratification1']].copy()  # Extracting relevant columns
    
    print(cardio_df1.columns, "\n")  

    # Plotting pie charts of features against target
    for i in cardio_df1:
        if i != 'data_value' and i != 'year' and i != 'locationabbr':
            plt.figure(figsize=(12,8))
            fig = px.pie(cardio_df1, names=i, values='data_value')
            fig.update_layout(title=f'Pie Chart of {i} against Cardiovascular Disease Rates')
            fig.show()

    # Handling missing values
    cardio_df1['data_value'] = cardio_df1['data_value'].astype(float)  # Converting data_value data type to float
    cardio_df1['data_value'] = cardio_df1.groupby(['locationabbr','year'])['data_value'].transform(lambda x: x.fillna(x.median()))  # Filling Nan with median
    print(cardio_df1.isnull().sum(), "\n")
    
    # Removing invalid year ranges
    cardio_df1 = cardio_df1[~cardio_df1['year'].str.contains('-')]

    return cardio_df1

# Asset for EDA of tobacco data
@asset
def eda_tobacco_data(load_data_from_mongodb):

    tobacco_df = load_data_from_mongodb["tobacco_df"]
    
    tobacco_df.drop(columns=['_id'], inplace=True)
    
    print(tobacco_df.info(), "\n")
    print(tobacco_df.shape, "\n")
    print(tobacco_df.isnull().sum(), "\n")

    # Checking for ranged years
    tobacco_df['year_new'] = tobacco_df['year'].apply(lambda x: 'Single year' if '-' not in x else 'Ranged year')
    count = tobacco_df['year_new'].value_counts()
    plt.figure(figsize=(12,8))
    count.plot(kind='barh', color=['cyan', 'yellow'], title='Single year vs Ranged years')
    plt.xlabel('Count')
    plt.ylabel('Year type')
    plt.show()

    tobacco_df1 = tobacco_df[['year', 'locationabbr', 'data_value', 'age', 'gender', 'race']].copy()
    
    print(tobacco_df1.columns, "\n")

    for i in tobacco_df1:
        if i != 'data_value' and i != 'year' and i != 'locationabbr':
            plt.figure(figsize=(12,8))
            fig = px.pie(tobacco_df1, names=i, values='data_value')
            fig.update_layout(title=f'Pie Chart of {i} against Tobacco Use Rates')
            fig.show()
            
    tobacco_df1['data_value'] = tobacco_df1['data_value'].astype(float)  
    tobacco_df1['data_value'] = tobacco_df1['data_value'].fillna(tobacco_df1['data_value'].median())  
    print(tobacco_df1.isnull().sum(), "\n")
    
    tobacco_df1 = tobacco_df1[~tobacco_df1['year'].str.contains('-')]

    return tobacco_df1

# Asset for EDA of obesity data
@asset
def eda_obesity_data(load_data_from_mongodb):

    obesity_df = load_data_from_mongodb["obesity_df"]

    obesity_df.drop(columns=['_id'], inplace=True)
    
    print(obesity_df.info(), "\n")
    print(obesity_df.shape, "\n")
    print(obesity_df.isnull().sum(), "\n")

    obesity_df1 = obesity_df[['yearstart', 'locationabbr', 'question', 'data_value', 'stratification1']].copy()
    
    print(obesity_df1.columns, "\n")

    for i in obesity_df1:
        if i != 'data_value' and i != 'yearstart' and i != 'locationabbr':
            plt.figure(figsize=(12,8))
            fig = px.pie(obesity_df1, names=i, values='data_value')
            fig.update_layout(title=f'Pie Chart of {i} against Obesity Rates')
            fig.show()

    obesity_df1['data_value'] = obesity_df1['data_value'].astype(float)  # Converting data_value data type to float
    obesity_df1['data_value'] = obesity_df1['data_value'].fillna(obesity_df1['data_value'].median())  # Filling missing values with median
    print(obesity_df1.isnull().sum(), "\n")
    
    obesity_df1 = obesity_df1[~obesity_df1['yearstart'].str.contains('-')]

    return obesity_df1 

# PostgreSQL connection string
engine = create_engine("postgresql+psycopg2://dap:dap@localhost:5432/postgres")

# Asset to store cardiovascular data to PostgreSQL
@asset
def store_cardio_data(eda_cardio_data):

    try:
        cardio_df1.to_sql('cardiovascular', engine, if_exists='replace', index=False)
        print("Cardiovascular data stored successfully.")
    except Exception as e:
        print(f"An error occurred during storing cardiovascular data: {e}")

# Asset to store tobacco data to PostgreSQL
@asset
def store_tobacco_data(eda_tobacco_data):

    try:
        tobacco_df1.to_sql('tobacco', engine, if_exists='replace', index=False)
        print("Tobacco use data stored successfully.")
    except Exception as e:
        print(f"An error occurred during storing tobacco use data: {e}")

# Asset to store obesity data to PostgreSQL
@asset
def store_obesity_data(eda_obesity_data):

    try:
        obesity_df1.to_sql('obesity', engine, if_exists='replace', index=False)
        print("Obesity data stored successfully.")
    except Exception as e:
        print(f"An error occurred during storing obesity data: {e}")

# Asset to retrieve cardiovascular data to PostgreSQL
@asset
def retrieve_cardio_data():

    try:
        cardio = pd.read_sql("SELECT * FROM cardiovascular", engine)
        print("Cardiovascular data retrieved successfully.")
        print(cardio.head())
        return cardio
    except Exception as e:
        print(f"An error occurred during retrieving cardiovascular data: {e}")

# Asset to retrieve tobacco data to PostgreSQL
@asset
def retrieve_tobacco_data():

    try:
        tobacco = pd.read_sql("SELECT * FROM tobacco", engine)
        print("Tobacco data retrieved successfully.")
        print(tobacco.head())
        return tobacco
    except Exception as e:
        print(f"An error occurred during retrieving tobacco data: {e}")

# Asset to retrieve obesity data to PostgreSQL
@asset
def retrieve_obesity_data():

    query = """
    SELECT * FROM obesity WHERE question IN (
        'Percent of adults aged 18 years and older who have an overweight classification',
        'Percent of adults aged 18 years and older who have obesity',
        'Percent of adults who engage in no leisure-time physical activity');
    """
    try:
        obesity = pd.read_sql(query, engine)
        print("Obesity data retrieved successfully.")
        print(obesity.head())
        return obesity
    except Exception as e:
        print(f"An error occurred during retrieving obesity data: {e}")

# Asset to merge cardiovascular, tobacco, and obesity data
@asset
def merge_data(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data):

    # Only selecting data from 2011-2019 from all three datasets
    query1 = """ 
    SELECT year, locationabbr, data_value FROM cardiovascular WHERE CAST(year AS INTEGER) BETWEEN 2011 AND 2019;
    """
    query2 = """ 
    SELECT year, locationabbr, data_value FROM tobacco WHERE CAST(year AS INTEGER) BETWEEN 2011 AND 2019;
    """
    query3 = """ 
    SELECT yearstart, locationabbr, data_value FROM obesity WHERE CAST(yearstart AS INTEGER) BETWEEN 2011 AND 2019;
    """
    
    # Retreiving data based on the queries
    cardio_merge = pd.read_sql(query1, engine)
    tobacco_merge = pd.read_sql(query2, engine)
    obesity_merge = pd.read_sql(query3, engine)
    
    print("Cardiovascular data for 2011-2019:\n", cardio_merge.head())
    print("Tobacco data for 2011-2019:\n", tobacco_merge.head())
    print("Obesity data for 2011-2019:\n", obesity_merge.head())
    print("Columns in cardiovascular_merge:", cardio_merge.columns)
    print("Columns in tobacco_merge:", tobacco_merge.columns)
    print("Columns in obesity_merge:", obesity_merge.columns)

    return {
            "cardio_merge": cardio_merge,
            "tobacco_merge": tobacco_merge,
            "obesity_merge": obesity_merge
        }

# Asset to merge cardiovascular and tobacco data
@asset
def merge_cardio_tobacco(merge_data):

    cardio_merge = merge_data["cardio_merge"]
    tobacco_merge = merge_data["tobacco_merge"]

    merged_dataset_ct = pd.merge(cardio_merge, tobacco_merge, on=['year', 'locationabbr'], how='outer')
    print("Merged Cardiovascular and Tobacco data:\n", merged_dataset_ct.head())
    return merged_dataset_ct

# Asset to merge cardiovascular and obesity data
@asset
def merge_cardio_obesity(merge_data):

    cardio_merge = merge_data["cardio_merge"]
    obesity_merge = merge_data["obesity_merge"]
    
    merged_dataset_co = pd.merge(cardio_merge, obesity_merge, left_on=['year', 'locationabbr'], right_on=['yearstart', 'locationabbr'], how='outer')
    print("Merged Cardiovascular and Obesity data:\n", merged_dataset_co.head())
    return merged_dataset_co

# Asset to merge tobacco and obesity data
@asset
def merge_tobacco_obesity(merge_data):

    tobacco_merge = merge_data["tobacco_merge"]
    obesity_merge = merge_data["obesity_merge"]

    merged_dataset_to = pd.merge(tobacco_merge, obesity_merge, left_on=['year', 'locationabbr'], right_on=['yearstart', 'locationabbr'], how='outer')
    print("Merged Tobacco and Obesity data:\n", merged_dataset_to.head())
    return merged_dataset_to

# Asset to handle missing values
@asset
def handle_missing_values(merge_cardio_tobacco, merge_cardio_obesity, merge_tobacco_obesity):

    merged_dataset_ct = merge_cardio_tobacco
    merged_dataset_co = merge_cardio_obesity
    merged_dataset_to = merge_tobacco_obesity

    merged_dataset_ct['data_value_x'] = merged_dataset_ct['data_value_x'].astype(float)  # Converting data_value data type to float
    merged_dataset_ct['data_value_y'] = merged_dataset_ct['data_value_y'].astype(float)
    merged_dataset_co['data_value_x'] = merged_dataset_co['data_value_x'].astype(float)
    merged_dataset_co['data_value_y'] = merged_dataset_co['data_value_y'].astype(float)
    merged_dataset_to['data_value_x'] = merged_dataset_to['data_value_x'].astype(float)
    merged_dataset_to['data_value_y'] = merged_dataset_to['data_value_y'].astype(float)

    merged_dataset_ct['data_value_x'] = merged_dataset_ct['data_value_x'].fillna(merged_dataset_ct['data_value_x'].median())  # Filling missing values with median
    merged_dataset_ct['data_value_y'] = merged_dataset_ct['data_value_y'].fillna(merged_dataset_ct['data_value_y'].median())
    merged_dataset_co['data_value_x'] = merged_dataset_co['data_value_x'].fillna(merged_dataset_co['data_value_x'].median())
    merged_dataset_co['data_value_y'] = merged_dataset_co['data_value_y'].fillna(merged_dataset_co['data_value_y'].median())
    merged_dataset_to['data_value_x'] = merged_dataset_to['data_value_x'].fillna(merged_dataset_to['data_value_x'].median())
    merged_dataset_to['data_value_y'] = merged_dataset_to['data_value_y'].fillna(merged_dataset_to['data_value_y'].median())
    merged_dataset_co['year'] = merged_dataset_co['year'].fillna(method='bfill')
    merged_dataset_to['year'] = merged_dataset_to['year'].fillna(method='bfill')

    print("After imputation:")
    print("Cardio-Tobacco merged missing values:\n", merged_dataset_ct.isnull().sum(), "\n")
    print("Cardio-Obesity merged missing values:\n", merged_dataset_co.isnull().sum(), "\n")
    print("Tobacco-Obesity merged missing values:\n", merged_dataset_to.isnull().sum(), "\n")

    return {
            "merged_dataset_ct": merged_dataset_ct,
            "merged_dataset_co": merged_dataset_co,
            "merged_dataset_to": merged_dataset_to
        }        

# Asset to show descriptive statistics
@asset
def descriptive_statistics(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data):

    cardio = retrieve_cardio_data
    tobacco = retrieve_tobacco_data
    obesity = retrieve_obesity_data
    
    print("Cardiovascular: \n", cardio.describe(), "\n")
    print("Tobacco: \n", tobacco.describe(), "\n")
    print("Obesity: \n", obesity.describe())

# Asset to show trend analysis
@asset
def trend_analysis(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data):
    
    cardio = retrieve_cardio_data
    tobacco = retrieve_tobacco_data
    obesity = retrieve_obesity_data
    
    cardio1 = cardio.copy()
    tobacco1 = tobacco.copy()
    obesity1 = obesity.copy()

    cardio1['year'] = pd.to_numeric(cardio1['year'], errors='coerce') 
    tobacco1['year'] = pd.to_numeric(tobacco1['year'], errors='coerce') 
    obesity1['yearstart'] = pd.to_numeric(obesity1['yearstart'], errors='coerce') 

    cardio_trend = cardio1.groupby('year')['data_value'].mean() # Calculating average for trend
    tobacco_trend = tobacco1.groupby('year')['data_value'].mean()
    obesity_trend = obesity1.groupby('yearstart')['data_value'].mean()

    def trends(data, trend, title, x_axis, y_axis, color, plot_bgcolor, paper_bgcolor, template="seaborn"):  
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend.index,
            y=trend.values,
            mode='lines+markers',
            name=title,
            line=dict(color=color),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template=template,
            hovermode="x unified",
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor
        )
        fig.show()

    trends(cardio1, cardio_trend, "Cardiovascular Disease trend", "Year", "Average rate", "red", plot_bgcolor="rgba(255, 200, 200, 1)",paper_bgcolor="rgba(255, 240, 240, 1)")
    trends(tobacco1, tobacco_trend, "Tobacco use trend", "Year", "Average rate", "blue", plot_bgcolor="rgba(173, 216, 230, 1)", paper_bgcolor="rgba(240, 248, 250, 1)")
    trends(obesity1, obesity_trend, "Obesity trend", "Year", "Average rate", "green", plot_bgcolor="rgba(144, 238, 144, 1)", paper_bgcolor="rgba(245, 255, 250, 1)")

# Asset to show location trend analysis
@asset
def location_trend_analysis(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data):
    
    cardio = retrieve_cardio_data
    tobacco = retrieve_tobacco_data
    obesity = retrieve_obesity_data
    
    cardio1 = cardio.copy()
    tobacco1 = tobacco.copy()
    obesity1 = obesity.copy()

    cardio_geo = cardio1.groupby('locationabbr', as_index=False)['data_value'].mean()
    tobacco_geo = tobacco1.groupby('locationabbr', as_index=False)['data_value'].mean()
    obesity_geo = obesity1.groupby('locationabbr', as_index=False)['data_value'].mean()

    us_states = px.data.election_geojson()

    def create_choropleth_map(data, us_states, loc_col='locationabbr', color_col='data_value', title='Rates by location', labels=None):
        fig = px.choropleth(
            data,
            locations=loc_col,
            color=color_col,
            geojson=us_states,
            locationmode='USA-states',  
            title=title,
            labels=labels
        )
        fig.show()

    create_choropleth_map(cardio_geo, us_states, loc_col='locationabbr', color_col='data_value', title='Cardiovascular disease rates by Location', labels={'data_value': 'Disease Rates', 'locationabbr': 'Location Abbreviation'})
    create_choropleth_map(tobacco_geo, us_states, loc_col='locationabbr', color_col='data_value', title='Tobacco usage rates by Location', labels={'data_value': 'Tobacco Rates', 'locationabbr': 'Location Abbreviation'})
    create_choropleth_map(obesity_geo, us_states, loc_col='locationabbr', color_col='data_value', title='Obesity rates by Location', labels={'data_value': 'Obesity Rates', 'locationabbr': 'Location Abbreviation'})

# Asset to perform regression analysis
@asset
def regression_analysis(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data):
    from statsmodels.formula.api import ols

    cardio = retrieve_cardio_data
    tobacco = retrieve_tobacco_data
    obesity = retrieve_obesity_data

    # Regression analysis for cardiovascular disease death rates
    cardio1 = cardio.copy()
    model_cardio = ols('data_value ~ C(year) + locationabbr + C(stratification1)', data=cardio1)
    fit_cardio = model_cardio.fit()
    print(fit_cardio.summary())

    # Regression analysis for tobacco use rates
    tobacco1 = tobacco.copy()
    model_tobacco = ols('data_value ~ C(year) + locationabbr + C(age) + C(race) + C(gender)', data=tobacco1)
    fit_tobacco = model_tobacco.fit()
    print(fit_tobacco.summary())

    # Regression analysis for obesity rates
    obesity1 = obesity.copy()
    model_obesity = ols('data_value ~ C(yearstart) + question + locationabbr + C(stratification1)', data=obesity1)
    fit_obesity = model_obesity.fit()
    print(fit_obesity.summary())

# Asset to perform correlation analysis
@asset
def correlation_analysis(retrieve_cardio_data, retrieve_tobacco_data, retrieve_obesity_data, handle_missing_values):

    cardio = retrieve_cardio_data
    tobacco = retrieve_tobacco_data
    obesity = retrieve_obesity_data

    merged_dataset_ct = handle_missing_values["merged_dataset_ct"]
    merged_dataset_co = handle_missing_values["merged_dataset_co"]
    merged_dataset_to = handle_missing_values["merged_dataset_to"]
    
    cardio1 = cardio.copy()
    tobacco1 = tobacco.copy()
    obesity1 = obesity.copy()
    
    merged_ct = merged_dataset_ct.copy()
    merged_co = merged_dataset_co.copy()
    merged_to = merged_dataset_to.copy()

    # Calculating correlations 
    ct_cor = cardio1[['year']].corrwith(tobacco1['data_value'])
    co_cor = cardio1[['year']].corrwith(obesity1['data_value'])
    tc_corr = tobacco1[['year']].corrwith(cardio1['data_value'])
    to_corr = tobacco1[['year']].corrwith(obesity1['data_value'])
    oc_corr = obesity1[['yearstart']].corrwith(cardio1['data_value'])
    ot_corr = obesity1[['yearstart']].corrwith(tobacco1['data_value'])

    print(ct_cor, "\n\n", co_cor, "\n\n", tc_corr, "\n\n", to_corr, "\n\n", oc_corr, "\n\n", ot_corr)

    merged_corr_ct = merged_ct[['data_value_x', 'data_value_y']].corr()
    merged_corr_co = merged_co[['data_value_x', 'data_value_y']].corr()
    merged_corr_to = merged_to[['data_value_x', 'data_value_y']].corr()

    print("\n")
    print(merged_corr_ct, "\n\n", merged_corr_co, "\n\n", merged_corr_to)

    # Heatmap for correlation
    sns.heatmap(merged_corr_ct, annot=True, cmap='coolwarm')
    plt.title("Cardiovascular disease rates against tobacco rates")
    plt.show()
    sns.heatmap(merged_corr_co, annot=True, cmap='coolwarm')
    plt.title("Cardiovascular disease rates against obesity rates")
    plt.show()
    sns.heatmap(merged_corr_to, annot=True, cmap='coolwarm')
    plt.title("Tobacco rates against Obesity rates")
    plt.show()

# Asset to perform chi-square analysis
@asset
def chi_square_analysis(handle_missing_values):
    
    from scipy.stats import chi2_contingency

    merged_dataset_ct = handle_missing_values["merged_dataset_ct"]
    merged_dataset_co = handle_missing_values["merged_dataset_co"]
    merged_dataset_to = handle_missing_values["merged_dataset_to"]

    merged_ct = merged_dataset_ct.copy()
    merged_co = merged_dataset_co.copy()
    merged_to = merged_dataset_to.copy()

    contigency_table_ct = pd.crosstab(merged_ct['locationabbr'], merged_ct['data_value_y'])
    chi2, p, dof, expected = chi2_contingency(contigency_table_ct)
    print(f"Chi-square for cardio location and tobacco rates: {p}")

    contigency_table_co = pd.crosstab(merged_co['locationabbr'], merged_co['data_value_y'])
    chi2, p, dof, expected = chi2_contingency(contigency_table_co)
    print(f"Chi-square for cardio location and obesity rates: {p}")

    contigency_table_to = pd.crosstab(merged_to['locationabbr'], merged_to['data_value_y'])
    chi2, p, dof, expected = chi2_contingency(contigency_table_to)
    print(f"Chi-square for tobacco location and obesity rates: {p}")

# Function to run the Dash app
@asset
def Dashboard():
    print("Dashboard in another script")
