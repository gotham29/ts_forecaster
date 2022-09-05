import os
import sys
import time
import operator
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import (
    mape,
    mase,
    mse,
    rmse,
    mae,
    ope,
    marre,
    r2_score,
    dtw_metric)
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis, plot_hist
from darts.models import (
    VARIMA,
    NBEATSModel,
    TCNModel,
    TransformerModel,
    RNNModel,
    LightGBMModel
)

METRICNAMES_METRICS = {
    'mape': mape,
    'mase': mase,
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'ope': ope,
    'marre': marre,
    'r2_score': r2_score,
    'dtw_metric': dtw_metric,
}

MODNAMES_MODELS = {
    'VARIMA':VARIMA,
    'RNNModel':RNNModel,
    'NBEATSModel':NBEATSModel,
    'TCNModel':TCNModel,
    'TransformerModel':TransformerModel,
    'LightGBMModel':LightGBMModel,
}

MODNAMES_LAGPARAMS = {
    'VARIMA':'p',
    'RNNModel':'input_chunk_length',
    'NBEATSModel':'input_chunk_length',
    'TCNModel':'input_chunk_length',
    'TransformerModel':'input_chunk_length',
    'LightGBMModel':'lags',
}

EVALS_COMPARES = {
    'mae':operator.lt,
    'mse':operator.lt,
    'rmse':operator.lt,
    'mape':operator.lt,
    'mase':operator.lt,
    'ope':operator.lt,
    'marre':operator.lt,
    'dtw_metric':operator.lt,
    'r2_score':operator.gt,
}


def train_save_models(data_dict:dict, modnames_grids: dict, dir_models: str, time_col: str, eval_metric: str, forecast_horizon: int):
    print(f'Training {len(modnames_grids)} models...')
    # Convert df to darts timeseries
    darts_timeseries = TimeSeries.from_dataframe(df=data_dict['t0'], time_col=time_col)
    # Train all models
    modnames_models, modnames_params, modnames_scores = {}, {}, {}
    for mod_name, mod_grid in modnames_grids.items():
        print(f"  mod_name = {mod_name}")
        model_untrained, params, score = gridsearch_model(model=MODNAMES_MODELS[mod_name],
                                                            mod_name=mod_name,
                                                            mod_grid=mod_grid,
                                                            data_t0=darts_timeseries,
                                                            eval_metric=eval_metric,
                                                            forecast_horizon=forecast_horizon,
                                                            time_col=time_col,
                                                            verbose=False)
        modnames_models[mod_name] = model_untrained.fit(series=darts_timeseries)
        modnames_params[mod_name] = params
        modnames_scores[mod_name] = score
        path_out = os.path.join(dir_models, f"{mod_name}.pkl")
        modnames_models[mod_name].save(path_out)
        print(f"    params = {params}\n    score = {score}")
    return modnames_models, modnames_params, modnames_scores


def get_eval(ts_pred, ts_true, time_col, eval_metric):
    # actuals = TimeSeries.from_dataframe(df, time_col=time_col)
    eval_metric = METRICNAMES_METRICS[eval_metric](actual_series=ts_true, pred_series=ts_pred)
    return eval_metric


def get_model_lag(mod_name, model):
    lag_param = MODNAMES_LAGPARAMS[mod_name]
    return model.model_params[lag_param]


def get_preds_rolling(model, df, features, LAG, time_col, forecast_horizon):
    feats = list(features)+[time_col]
    preds = []
    for _ in range(df.shape[0]):
        if _ < LAG:
            continue
        df_lag = df[_-LAG:_][ feats ]
        ts = TimeSeries.from_dataframe(df_lag, time_col=time_col) #df_row
        pred = model.predict(n=forecast_horizon, series=ts)
        preds.append(pred.data_array().values.reshape( len(features) ))
    return np.array(preds)


def get_modnames_preds(modnames_models, df, time_col, forecast_horizon, LAG_MIN=3):
    print('Getting modnames_preds...')
    modnames_preds = {}
    # Get preds -- rolling forward in time
    for mod_name, model in modnames_models.items():
        preds = get_preds_rolling(model=model,
                                    df=df,
                                    features=model.training_series.components,
                                    LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
                                    time_col=time_col,
                                    forecast_horizon=forecast_horizon)
        df_preds = pd.DataFrame(preds, columns=features)
        time_vals = df[time_col].values[-df_preds.shape[0]:]
        df_preds.insert(0, time_col, time_vals)
        modnames_preds[mod_name] = df_preds
    print('  --> done')
    return modnames_preds


def get_modnames_evals(modnames_preds, df_true, time_col, eval_metric):
    print('Getting modnames_evals...')
    modnames_evals = {}
    for mod_name, preds in modnames_preds.items():
        ts_pred = TimeSeries.from_dataframe(preds, time_col=time_col)
        ts_true = TimeSeries.from_dataframe(df_true.tail(len(preds)), time_col=time_col)
        modnames_evals[mod_name] = get_eval(ts_pred, ts_true, time_col, eval_metric)
    print('  --> done')
    return modnames_evals


def gridsearch_model(model, mod_name, mod_grid, data_t0, forecast_horizon, eval_metric, time_col=None, verbose=True):
    model_best = model.gridsearch(parameters=mod_grid,
                                    series=data_t0,  #darts.TimeSeries
                                    forecast_horizon=forecast_horizon,
                                    verbose=verbose,
                                    metric=METRICNAMES_METRICS[eval_metric])
    return model_best


def get_model_best(modnames_scores, eval_metric):
    print("Getting best model...")
    # Get compare function for eval_metric
    comp = EVALS_COMPARES[eval_metric]
    print(f"  eval = {eval_metric}\n  comp = {comp}")
    # Get best model
    best_metric, best_mod = np.inf, None
    for mod_name, current_metric in modnames_scores.items():
        if comp(current_metric, best_metric): #current_metric vs best_metric:
            best_metric = current_metric
            best_mod = mod_name
            print(f"  {best_metric} <-- {best_mod}")
    print(f"    *{best_mod}*")
    return best_mod


def save_results(modnames_params: dict, modnames_scores: dict, dir_out: str, name: str, eval_metric:str):
    results = {'name': [], 'params':[], 'eval':[]}
    for mod_name, mod_params in modnames_params.items():
        results['name'].append(mod_name)
        results['params'].append(mod_params)
        results['eval'].append(modnames_scores[mod_name])
    ascending = True
    if EVALS_COMPARES[eval_metric] == operator.gt:
        ascending = False
    results = pd.DataFrame(results).sort_values(by='eval', ascending=ascending)
    path_out = os.path.join(dir_out, f'{name}.csv')
    results.to_csv(path_out, index=False)