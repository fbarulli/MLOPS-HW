from asyncio.log import logger
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task , context, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

from datetime import datetime, timedelta
from dateutil.relativedelta import *

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

#Q2. Parameterizing the flow
@task()
def get_paths(date=None):
    

    if date is None:
        
        today_date = datetime.now()
        train_month = (today_date - relativedelta(months=2)).strftime(format='%Y-%m')
        val_month = (today_date - relativedelta(months=1)).strftime(format='%Y-%m')
        train_path = f'./home/main-user/notebooks/data/fhv_tripdata_{train_month}.parquet'
        val_path = f'./home/main-user/notebooks/data/fhv_tripdata_{val_month}.parquet'


    else:
        date_str_format = datetime.strptime(date, '%Y-%m-%d')
        train_month = (date_str_format - relativedelta(months=2)).strftime(format='%Y-%m')
        
        val_path = f'./home/main-user/notebooks/data/fhv_tripdata_{val_month}.parquet'
        train_path = f'./home/main-user/notebooks/data/fhv_tripdata_{train_month}.parquet'
        pass
    return train_path, val_path




@task()
def prepare_features(df, categorical, train=True):
    logger  = get_run_logger()
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime	
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        #print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")

    else:
        #print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task()
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    #print(f"The shape of X_train is {X_train.shape}")
    #print(f"The DictVectorizer has {len(dv.feature_names_)} features")


    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    #print(f"The MSE of training is: {mse}")
    return lr, dv


@task()
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    #print(f"The MSE of validation is: {mse}")
    return


'''
The main function will be converted to a flow and the other functions will be tasks. 
After adding all of the decorators, there is actually one task that you will need to 
call .result() for inside the flow to get it to work. Which task is this?
'''




@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    train_path, val_path = get_paths(date).result()

    categorical = ['PULocationID', 'DOLocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)
    # Save model, dv
    with open(f'/home/main-user/mlops-zoomcamp/models/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)
    with open(f'/home/main-user/mlops-zoomcamp/models/dv/dv-{date}.b', 'wb') as f_out:
        pickle.dump(dv, f_out)



DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["Fabi-ml"]
)  
