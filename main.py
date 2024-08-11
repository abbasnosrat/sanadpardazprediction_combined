import datetime
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jdatetime
from matplotlib.style import use
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from tqdm.auto import trange
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_percentage_error

from utils import *












st.write("""
         # Sales Prediction
         Hover your cursor on the ? if you want information on each component. Also, the documentation is available on [this Google doc](https://docs.google.com/document/d/1oMk5kQi6FAgqsGGXW-ksRVP8OyhvmnbUnxn0mpi5x2U/edit?usp=sharing). You can find a detailed guide of the app on [this doc](https://docs.google.com/document/d/1J3bzPC_u5nAXrmgdaiQtL9J35yV_dVR7XLDImyE_78Y/edit?usp=sharing)
         """)
st.sidebar.write("Controls")
sheet_id = "1PNTC8IvqruHs3DWVX6HW30d2TCM6z3PCxtRMA_qep0M"
sheet_name = "Sheet1"
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
helps = pd.read_csv(url,index_col=0)
file = st.sidebar.file_uploader("Upload Your Dataset", type=".xlsx",help=helps.loc["Upload Your Dataset"].Description)
use_sample_data = st.sidebar.checkbox("Use Sample Data",
                                      help=helps.loc["Use Sample Data"].Description)

# df = pd.read_csv("SalesData.csv") if file is None else pd.read_csv(file)
try:
    df = pd.read_excel(file)
    got_data = True
except Exception as e:
    print(e)
    if use_sample_data:
        df = pd.read_csv("./SaleData.csv") 
        got_data = True
    else:
        got_data = False

if got_data:
    df["Year"] = df["StrFactDate"].apply(lambda d: int(d.split("/")[0]))
    df["Month"] = df["StrFactDate"].apply(lambda d: int(d.split("/")[1]))
    df = df.groupby(["GoodName","Year", "Month"]).agg({"SaleAmount":"sum"}).reset_index()
    df_v = pd.DataFrame(df["GoodName"].value_counts()).reset_index()
    
    df_v.columns = ["GoodName", "Count"]
    df_v = df_v.query("Count > 6")
    products = list(df_v.GoodName.unique())
    product = st.sidebar.selectbox(label="Please select a Product", options=products,
                                    help=helps.loc["Please Select A Product"].Description)
    df_t = df.query(f"GoodName == '{product}'").reset_index(drop=True).copy()

    
   

    df_t = df_t.sort_values(["Year", "Month"], ignore_index=True).drop(columns="GoodName")
        
        
    horizon = int(st.sidebar.slider(label="Select Prediction Horizon", min_value=2, max_value=30, value=5,
                                    help=helps.loc["Select Prediction Horizon"].Description))
    test_size_manual = st.sidebar.number_input(label="Select Test Size", min_value=0, max_value=30, value=0,
                                               help=helps.loc["Select Test Size"].Description)
    manual = st.sidebar.checkbox("Manual Mode", help=helps.loc["Manual Mode"].Description)
    

    



    train_size = -5 if test_size_manual == 0 else -test_size_manual
    model_name = st.selectbox("select your model", options=["XGB", "Prophet"])
    if model_name == "XGB":
        XGB(manual, df_t, train_size, horizon, helps)
    else:
        FBProphet(manual, df_t,test_size_manual, horizon, helps)

    # plt.plot(preds, label="prediction")
    # plt.savefig("ar_pred.jpg")
else:
    
    st.write("Please upload your data")
    # df = pd.read_csv("SalesData.csv")[["GoodName", "StrFactDate", "SaleAmount"]]
    # csv = convert_df(df)
    # st.download_button("Sample Data", csv, "SampleData.csv","text/csv",
    # key='download-csv')