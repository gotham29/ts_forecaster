import os
import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from darts.dataprocessing.transformers import Scaler
from darts.utils.statistics import stationarity_tests, stationarity_test_adf, stationarity_test_kpss


SCALETYPES_SCALERS = {
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'standard': StandardScaler,
    }


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
    t0, t1 = data[:t0_endrow][features_all], data[t0_endrow:][features_all]
    x_t0, y_t0 = t0[features_inout['in']+[time_col]], t0[features_inout['pred']+[time_col]]
    x_t1, y_t1 = t1[features_inout['in']+[time_col]], t1[features_inout['pred']+[time_col]]
    dict_data = {'x_t0':x_t0, 'x_t1':x_t1, 'y_t0':y_t0, 'y_t1':y_t1, 't0':t0, 't1':t1, 't0t1': pd.concat([t0,t1], axis=0)}
    if train_models:
        for dname, d in dict_data.items():
            print(f"  {dname} = {d.shape}")
    return dict_data


def check_stationarity(df, time_col, output_dir,
                        tests_nulls = {'ADFULLER': '(null=stationary)',
                                        'KPSS': '(null=stationary)'},
                        tests_functions = {'ADFULLER':stationarity_test_adf,
                                            'KPSS':stationarity_test_kpss}):
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
    Outputs:
        n/a (csv saved)
    """
    path_out = os.path.join(output_dir,'stationary_tests.csv')
    cols_pvals = {c:None for c in df if c != time_col}
    tests_colspvals = {t:cols_pvals for t in tests_functions}
    for test, cols_pvals in tests_colspvals.items():
        for col, pvaldict in cols_pvals.items():
            ts = TimeSeries.from_dataframe(df[[col, time_col]], time_col=time_col)
            is_stat = tests_functions[test](ts=ts)
            p_val = is_stat[1]
            tests_colspvals[test][col] = f"p={p_val}"
    tests_colspvals = {f"{test}\n  {tests_nulls[test]}": cols_pvals for test, cols_pvals in tests_colspvals.items()}
    df_tests_colspvals = pd.DataFrame(tests_colspvals)
    df_tests_colspvals.to_csv(path_out)


def scale_data(data, features, scale_type=False, scaler=False, rescale=False):
    """
    Purpose:
        Scale/rescale data
    Inputs:
        data:
            type: pd.DataFrame
            meaning: data to to scaled/rescaled
        features:
            type: list
            meaning: features to be scaled/rescaled
        scaler:
            type: sklearn.preprocessing OR bool
            meaning: scaler function (if bool scaler is create from 'scale_type')
        scale_type:
            type: str
            meaning: type of sklearn scaler to use (looked up in SCALETYPES_SCALERS)
        rescale:
            type: bool
            meaning: rescale instead of scale (default=False)
    Outputs:
        data:
            type: darts.TimeSeries
            meaning: scaled/rescaled data
        scaler:
            type: sklearn.preprocessing
            meaning: scaler function
    """
    data_ts = TimeSeries.from_dataframe(data[features])
    if not scaler:
        assert scale_type != False, "scale_type must be provided if scaler obj is not"
        scaler = SCALETYPES_SCALERS[scale_type](feature_range=(-1, 1))
    transformer = Scaler(scaler)
    if rescale:
        data_ = transformer.inverse_transform(data_ts)
    else:  #scale
        data_ = transformer.fit_transform(data_ts)
    return data, scaler
