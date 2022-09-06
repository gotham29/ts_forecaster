import os
import pandas as pd
from darts import TimeSeries

def split_data(data, data_cap: int, time_col: str, features_inout: dict, test_prop: float, train_models: bool):
    """
    Purpose:
        Split data to x/y, t0/t1
    Inputs:
        data:
            type: pd.DataFrame
            meaning: data to split to x/y, t0/t1
        data_cap:
            type: int
            meaning: max number of data rows to use (first n)
        time_col:
            type: str
            meaning: name of timestamp column
        features_inout:
            type: dict
            meaning: keys are feature types ('in' and 'pred') values are feature lists
        test_prop:
            type: float
            meaning: proportion of data for test split
        train_models:
            type: bool
            meaning: whether to train models
    Outputs:
        dict_data:
            type: dict
            meaning: keys are data names, values are pd.DataFrames
    """
    data = data[:data_cap]
    train_prop = 1-test_prop
    train_split, test_split = int(train_prop*100), int(test_prop*100)
    if train_models:
        print(f'Spltting data Time0 / Time1 by {train_split} / {test_split}...')
    t0_endrow = int(data.shape[0]*train_prop)
    features_all = list(set(features_inout['in']+features_inout['pred']))
    if time_col:
        features_all += [time_col]
    # TODO: add time_col
    t0, t1 = data[:t0_endrow][features_all], data[t0_endrow:][features_all]
    x_t0, y_t0 = t0[features_inout['in']+[time_col]], t0[features_inout['pred']+[time_col]]
    x_t1, y_t1 = t1[features_inout['in']+[time_col]], t1[features_inout['pred']+[time_col]]
    dict_data = {'x_t0':x_t0, 'x_t1':x_t1, 'y_t0':y_t0, 'y_t1':y_t1, 't0':t0, 't1':t1, 't0t1': pd.concat([t0,t1], axis=0)}
    if train_models:
        for dname, d in dict_data.items():
            print(f"  {dname} = {d.shape}")
    return dict_data


def check_stationarity(df, time_col, output_dir, p_vals=[0.01, 0.05, 0.10]):
    """
    Purpose:
        Check stationarity for all pred features
    Inputs:
        df:
            type: pd.DataFrame
            meaning: data to check stationary on
        time_col:
            type: str
            meaning: name of timestamp column
        output_dir:
            type: str
            meaning: dir to save outputs too
        p_vals:
            type: list of floats
            meaning: p_value thresholds
    Outputs:
        n/a (csv saved)
    """
    path_out = os.path.join(output_dir,'stationary_tests.csv')
    df_dict = {c:{f"p={p}":None for p in p_vals} for c in df if c != time_col}
    for col, pvaldict in df_dict.items():
        for p in p_vals:
            ts = TimeSeries.from_dataframe(df[[col, time_col]], time_col=time_col)
            is_stationary = stationarity_tests(ts, p_value_threshold_adfuller=p, p_value_threshold_kpss=p)
            df_dict[col][f"p={p}"] = is_stationary
    df = pd.DataFrame(df_dict)
    df.to_csv(path_out)

