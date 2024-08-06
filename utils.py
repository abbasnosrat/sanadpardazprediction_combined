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









@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')



def make_ticks(year,month):
    ticks = []
    for i in range(len(year)):
        ticks.append(f"{int(year[i])}/{int(month[i])}")
    return np.array(ticks)

def train_booster(df_t, params, train_size=-5):

    dg = df_t.copy().rename(columns={"SaleAmount":"y"})
    
    for i in range(int(params["n_lags"])):
   
        dg[f"y_lag_{i+1}"] = dg["y"].shift((i+1))
    dg = dg.fillna(0)

    y = dg["y"].to_numpy()
    X = dg.drop(columns="y").to_numpy()
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    param_dist = {
            'max_depth': stats.randint(1, 100),
            'learning_rate': stats.uniform(0.01, 0.1),
            'subsample': stats.uniform(0.5, 0.5),
            'n_estimators':stats.randint(1, 200)
        }

        # Create the XGBoost model object
    xgb_model = xgb.XGBRegressor()

        # Create the RandomizedSearchCV object
    # random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=30, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)

        # Fit the RandomizedSearchCV object to the training data
    xgb_model.fit(X_train, y_train)

    pred = xgb_model.predict(X_test)
    
    mse= mean_absolute_percentage_error(y_test, pred)
    
    return xgb_model, mse



def ar_prediction(model:XGBRegressor, X_init:np.ndarray, horizon:int):
    Xs = []
    preds = []
    year = np.array(X_init[0]).reshape(1)
    month = np.array(X_init[1]).reshape(1)
    if len(X_init.flatten()) >2:
        print(X_init)
        X_init = X_init.copy()
        # print(X_init)
        
        lags = X_init[2:]
        for i in range(horizon):
            X_pred = np.concatenate([year, month, lags]).reshape([1,-1])
            # print(X_pred)
            pred = model.predict(X_pred)
            Xs.append(X_pred.copy())
            preds.append(pred.copy())
            lags = np.concatenate([lags[1:], np.array(pred).reshape(-1)])
            month += 1
            if month > 12:
                month = np.array([1]).reshape(1)
                year += 1
                
        return np.concatenate(Xs, axis=0), np.concatenate(preds).reshape(-1)
    else:
        for i in range(horizon):
            print(year, month)
            X_pred = np.array([year, month]).reshape([1,-1])
            # print(X_pred)
            pred = model.predict(X_pred)
            Xs.append(X_pred.copy())
            preds.append(pred.copy())
            month += 1
            if month > 12:
                month = np.array([1]).reshape(1)
                year += 1
        return np.concatenate(Xs, axis=0), np.concatenate(preds).reshape(-1)




def XGB(manual, df_t, train_size, horizon, helps):
    

    if not manual:
        mses = []
        models = []
        lags  = range(10)
        # lags.set_description("Tuning the model. Please be patient")
        for lag in lags:
            
            model, mse = train_booster(df_t, {"n_lags": lag}, train_size)
            mses.append(mse)
            models.append(model)
        
        best = {"n_lags": np.argmin(mses)}
        # best_params = models[int(best["n_lags"])].best_params_
    else:
        n_lags_manual=float(st.sidebar.number_input(label="Select Number of lags", value=1,
                                                min_value=0,
                                                max_value=1000,
                                                help=helps.loc["Select Number of lags"].Description))
    
        best = {"n_lags": n_lags_manual}
        # model, mse = train_booster(df_t, {"n_lags": n_lags_manual}, train_size)
        # best_params = model.best_params_







    n_lags = int(best["n_lags"])

  

    dg = df_t.copy().rename(columns={"SaleAmount":"y"})
    for i in range(n_lags):
        dg[f"y_lag_{i+1}"] = dg["y"].shift((i+1))

    dg = dg.fillna(0)
    y = dg["y"].to_numpy()
    X = dg.drop(columns="y").to_numpy()
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # model = XGBRegressor(**best_params)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X)
    pred = np.maximum(0,pred)
    ticks = make_ticks(X[:,0], X[:,1])
    fig_tuned = go.Figure()
    fig_tuned.add_trace(go.Scatter(x=ticks, y=y, mode='markers', name='observations', marker=dict(color='black')))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred, mode='lines', name='predictions'))

    fig_tuned.add_vline(x=ticks[train_size], line=dict(dash='dash', color='black'), name='split')

    fig_tuned.update_layout(title='One Step Ahead Prediction', xaxis_title='Date', yaxis_title='SaleAmount')
    st.markdown("""
        <style>
        .chart-container {
            position: relative;
            width: 100%;
        }
        .help-icon {
            position: absolute;
            top: 10px;
            right: 10px;  /* Positioned to the right */
            font-size: 24px;
            cursor: pointer;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
       
    """ + f""" <div class="chart-container">
            <div class="tooltip">
                <span class="help-icon">❓</span>
                <span class="tooltiptext">{helps.loc['One Step Ahead Prediction'].Description}</span>
            </div>
        </div>""", unsafe_allow_html=True)    
    
    st.plotly_chart(fig_tuned)
    # plt.plot(ticks,y, ".k", label="observations")
    # plt.plot(pred, label="prediction")

    # plt.axvline(len(y) + train_size, linestyle ="--", color="k",label="Split")

    # plt.xticks(ticks, rotation=270)
    # plt.title(f"n_lags = {n_lags}")
    # plt.legend()
    # plt.savefig("ost_pred.jpg")
    
    # model = XGBRegressor(**best_params)
    model = XGBRegressor()
    model.fit(X, y)
    Xs, preds = ar_prediction(model, X[-1], horizon)
    preds = np.maximum(0, preds)
    ticks = make_ticks(Xs[:,0], Xs[:,1])
    fig_ar = go.Figure()
    fig_ar.add_trace(go.Scatter(x=ticks, y=preds, mode='lines', name='predictions'))
    fig_ar.update_layout(title='AutoRegressive Prediction', xaxis_title='Date', yaxis_title='Sales Amount')
    st.markdown("""
        <style>
        .chart-container {
            position: relative;
            width: 100%;
        }
        .help-icon {
            position: absolute;
            top: 10px;
            right: 10px;  /* Positioned to the right */
            font-size: 24px;
            cursor: pointer;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
       
    """ + f""" <div class="chart-container">
            <div class="tooltip">
                <span class="help-icon">❓</span>
                <span class="tooltiptext">{helps.loc['AutoRegressive Prediction'].Description}</span>
            </div>
        </div>""", unsafe_allow_html=True) 
    st.plotly_chart(fig_ar)
    # plt.figure()
    df_pred = pd.DataFrame({"Date":ticks, "Yhat":preds})
    
    csv = convert_df(df_pred)
    st.download_button(
    "Download Results",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv', help="Download the data behind the chart above in CSV format."
    )

def to_georgian(date):
    date  = date.split("/")
    date = jdatetime.date(year=int(date[0]), month=int(date[1]), day=int(date[2])).togregorian()
    return date

def to_jalali(row):
    d = jdatetime.date.fromgregorian(year=row.year, month=row.month, day=row.day)
    return f"{d.year}/{d.month}"

def FBProphet(manual, df_t,test_size_manual, horizon, helps):


    dg = df_t.copy()
    dg["ds"] = (dg["Year"].astype(str)+ "/"+dg["Month"].astype(str) + "/01").apply(to_georgian)
        
    dg = dg[["ds", "SaleAmount"]].rename(columns={"SaleAmount":"y"})
    # dg
    if test_size_manual == 0:
        train_size = -10 if len(dg)>20 else -2
    else:
        train_size = -test_size_manual
    ds_train = dg.iloc[:train_size]
    ds_test = dg.iloc[train_size:]
    if not manual:
        def hyperparameter_tuning(space):
            model = Prophet(
                            # changepoint_prior_scale=space["changepoint_prior_scale"],
                            # seasonality_prior_scale=space["seasonality_prior_scale"]
                            **space
                            
                            )
            model.fit(ds_train)
            future = pd.DataFrame(dg["ds"])
            future.index = future["ds"]
            future.drop("ds", axis=1)
            pred = model.predict(future)
            pred_data = pred.iloc[train_size:]
            mse = mean_absolute_percentage_error(ds_test["y"], pred_data["yhat"])
            return {'loss': mse, 'status': STATUS_OK, 'model': model}

        space = {
            'changepoint_prior_scale': hp.uniform("changepoint_prior_scale", 0.0001, 20),
            'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.001, 20)
        }

        trials = Trials()
        best = fmin(fn=hyperparameter_tuning,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=40,
                    trials=trials)
    else:
        changepoint_prior_scale_manual=float(st.sidebar.text_input(label="Select changepoint_prior_scale", value=1,
                                                                   help=helps.loc["Select changepoint_prior_scale"].Description))
        seasonality_prior_scale_manual=float(st.sidebar.text_input(label="Select seasonality_prior_scale", value=1,help=helps.loc["Select seasonality_prior_scale"].Description))
        best = {"changepoint_prior_scale":changepoint_prior_scale_manual,
                "seasonality_prior_scale":seasonality_prior_scale_manual}

    train_size = len(dg) + train_size

    model = Prophet(
        changepoint_prior_scale=best["changepoint_prior_scale"],
                    seasonality_prior_scale=best["seasonality_prior_scale"],
                    growth="linear")
    model.fit(ds_train)
    future = pd.DataFrame(dg["ds"])
    future.index = future["ds"]
    future.drop("ds", axis=1)
    pred = model.predict(future)

    ticks = dg["ds"].apply(to_jalali)
    
    st.markdown("""
        <style>
        .chart-container {
            position: relative;
            width: 100%;
        }
        .help-icon {
            position: absolute;
            top: 10px;
            right: 10px;  /* Positioned to the right */
            font-size: 24px;
            cursor: pointer;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
       
    """ + f""" <div class="chart-container">
            <div class="tooltip">
                <span class="help-icon">❓</span>
                <span class="tooltiptext">{helps.loc['Parameter Table'].Description}</span>
            </div>
        </div>""", unsafe_allow_html=True)    
    

    st.dataframe(best)

    fig_tuned = go.Figure()
    fig_tuned.add_trace(go.Scatter(x=ticks, y=dg["y"], mode='markers', name='observations', marker=dict(color='black')))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat"].apply(lambda y: np.maximum(0,y)), mode='lines', name='predictions'))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat_upper"].apply(lambda y: np.maximum(0,y)), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
    fig_tuned.add_trace(go.Scatter(x=ticks, y=pred["yhat_lower"].apply(lambda y: np.maximum(0,y)), fill='tonexty', mode='lines', line_color='lightgrey', showlegend=False))
    fig_tuned.add_vline(x=ticks[train_size], line=dict(dash='dash', color='black'), name='split')

    fig_tuned.update_layout(
        title='Tuned Model Predictions',
        xaxis_title='Date',
        yaxis_title='Sales Amount')



    st.markdown("""
        <style>
        .chart-container {
            position: relative;
            width: 100%;
        }
        .help-icon {
            position: absolute;
            top: 10px;
            right: 10px;  /* Positioned to the right */
            font-size: 24px;
            cursor: pointer;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        
    """ + f"""<div class="chart-container">
            <div class="tooltip">
                <span class="help-icon">❓</span>
                <span class="tooltiptext">{helps.loc['Tuned Model Predictions'].Description}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.plotly_chart(fig_tuned)

    model = Prophet(changepoint_prior_scale=best["changepoint_prior_scale"],
                seasonality_prior_scale=best["seasonality_prior_scale"],
                growth = "linear"
                )


    model.fit(dg)
    future = [dg["ds"].max() + datetime.timedelta(days= 30*(i+1)) for i in range(horizon)]
    future = pd.DataFrame({"ds": future})
    pred = model.predict(future)
    fig_final = go.Figure()
    ticks = pred["ds"].apply(to_jalali)
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat"].apply(lambda y: np.maximum(0,y)), mode='lines', name='predictions'))
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat_upper"].apply(lambda y: np.maximum(0,y)), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
    fig_final.add_trace(go.Scatter(x=ticks, y=pred["yhat_lower"].apply(lambda y: np.maximum(0,y)), fill='tonexty', mode='lines', line_color='lightgrey', showlegend=False))
    fig_final.update_layout(title=f'Predictions for {horizon} months', xaxis_title='Date', yaxis_title='Sales Amount')
    
    st.markdown("""
        <style>
        .chart-container {
            position: relative;
            width: 100%;
        }
        .help-icon {
            position: absolute;
            top: 10px;
            right: 10px;  /* Positioned to the right */
            font-size: 24px;
            cursor: pointer;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        <div class="chart-container">
            <div class="tooltip">
                <span class="help-icon">❓</span>
                <span class="tooltiptext">The chart displays the predictions for the user-defined prediction horizon. The model behind this chart is trained on the entire dataset.</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    
    
    st.plotly_chart(fig_final)
    df_final = pd.DataFrame({"Date": ticks, "Yhat":pred["yhat"].apply(lambda y: np.maximum(0,y)),
                             "Yhat_upper":pred["yhat_upper"].apply(lambda y: np.maximum(0,y)), 
                             "Yhat_lower": pred["yhat_lower"].apply(lambda y: np.maximum(0,y))})


    csv = convert_df(df_final)

    st.download_button(
    "Download Results",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv', help="Download the data behind the chart above in CSV format."
    )