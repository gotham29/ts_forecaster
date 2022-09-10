import os
import sys
import time
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    LightGBMModel,
)

LAG_MIN = 3

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
    'LightGBMModel':LightGBMModel
}

MODNAMES_LAGPARAMS = {
    'VARIMA':'p',
    'RNNModel':'input_chunk_length',
    'NBEATSModel':'input_chunk_length',
    'TCNModel':'input_chunk_length',
    'TransformerModel':'input_chunk_length',
    'LightGBMModel':'lags'
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


def train_save_models(data_dict:dict, do_gridsearch: bool, modnames_grids: dict, dir_models: str, time_col: str, eval_metric: str, forecast_horizon: int):
    """
    Purpose:
        Train and save all models found in config['modnames_grids']
    Inputs:
        data_dict:
            type: dict
            meaning: keys are data names ('x_t0', 'y_t0', 'x_t1', 'y_t1', 't0', 't1'), values are pd.DataFrames
        do_gridsearch:
            type: float
            meaning: whether to gridsearch for best params or use defaults
        modnames_grids:
            type: dict
            meaning: keys are model types, values are params grids to gridsearch
        dir_models:
            type: str
            meaning: dir to save models to
        time_col:
            type: str
            meaning: name of timestamp column
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
        forecast_horizon:
            type: int
            meaning: number of steps into the future to predict
    Outputs:
        modnames_models:
            type: dict
            meaning: keys are model types (i.e. 'VARIMA', 'LightGBMModel', etc), values are model objects
        modnames_params:
            type: dict
            meaning: keys are model types, values are best params (from gridsearch)
        modnames_scores:
            type: dict
            meaning: keys are model types, values are eval metric scores
    """
    print(f'Training {len(modnames_grids)} models; GridSearch = {do_gridsearch}')
    # Convert df to darts timeseries
    darts_timeseries = TimeSeries.from_dataframe(df=data_dict['t0'], time_col=time_col)
    # Train all models
    modnames_models, modnames_params, modnames_scores = {}, {}, {}
    for mod_name, mod_grid in modnames_grids.items():
        print(f"  mod_name = {mod_name}")
        if do_gridsearch:
            model_untrained, params, score = gridsearch_model(model=MODNAMES_MODELS[mod_name],
                                                                mod_name=mod_name,
                                                                mod_grid=mod_grid,
                                                                data_t0=darts_timeseries,
                                                                eval_metric=eval_metric,
                                                                forecast_horizon=forecast_horizon,
                                                                time_col=time_col,
                                                                verbose=False)
        else:
            params = slim_paramgrid(mod_grid)
            model_untrained = MODNAMES_MODELS[mod_name](**params)
            score = 0 ### no score since no gridsearch
        modnames_models[mod_name] = model_untrained.fit(series=darts_timeseries)
        modnames_params[mod_name] = params
        modnames_scores[mod_name] = score
        path_out = os.path.join(dir_models, f"{mod_name}.pkl")
        modnames_models[mod_name].save(path_out)
        print(f"    params = {params}\n    score = {score}")
    return modnames_models, modnames_params, modnames_scores


def slim_paramgrid(mod_grid, ind_select=0):
    """
    Purpose:
        Get 1 param set from mod_grid to init model (instead of gridsearch)
    Inputs:
        mod_grid:
            type: dict
            meaning: gridsearch grid from config['modnames_grids']
        ind_select:
            type: int (default=0)
            meaning: index of param list item to choose for slim
    Outputs:
        mg_slimmed:
            type: dict
            meaning: param grid slimmed from mod_grid -- ready to init model
    """
    mg_slimmed = {k:None for k in mod_grid}
    for param, val in mod_grid.items():
        if isinstance(val, list):
            val = val[ind_select]
        mg_slimmed[param] = val
    return mg_slimmed


def get_eval(ts_pred, ts_true, time_col, eval_metric):
    """
    Purpose:
        Get eval score for pred vs true from 'eval_metric' function
    Inputs:
        ts_pred:
            type: darts.Timeseries
            meaning: pred values
        ts_true
            type: darts.Timeseries
            meaning: true values
        time_col:
            type: str
            meaning: name of timestamp column
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
    Outputs:
        eval_score:
            type: float
            meaning: eval score for pred vs true from 'eval_metric' function
    """
    eval_score = METRICNAMES_METRICS[eval_metric](actual_series=ts_true, pred_series=ts_pred)
    return eval_score


def get_model_lag(mod_name, model):
    """
    Purpose:
        Get lag for given model (number of prior timesteps to pred from)
    Inputs:
        mod_name:
            type: str
            meaning: name of model type (i.e. 'VARIMA', 'LightGBMModel', etc)
        model:
            type: darts.ForecastingModel
            meaning: model object containing param values (including lag, which has is differently names for different model types, as held in MODNAMES_LAGPARAMS)
    Outputs:
        lag:
            type: int or list
            meaning: number of prior timesteps to pred from (for given model)
    """
    lag_param = MODNAMES_LAGPARAMS[mod_name]
    lag = model.model_params[lag_param]
    return lag


def get_preds_rolling(model, df, features, LAG, time_col, forecast_horizon):
    """
    Purpose:
        Get model predictions in rolling forward fashion
    Inputs:
        model:
            type: darts.ForecastingModel
            meaning: model object to get preds from
        df:
            type: pd.DataFrame
            meaning: data to run through model for preds
        features:
            type: list
            meaning: non-timestamp features used to train models
        LAG:
            type: int or list
            meaning: number of prior timesteps to pred from
        time_col:
            type: str
            meaning: name of timestamp column
        forecast_horizon:
            type: int
            meaning: number of steps into the future to predict
    Outputs:
        preds:
            type: np.array
            meaning: pred values for all pred features
    """
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
    """
    Purpose:
        Get preds from all models in 'modnames_models' arg
    Inputs:
        modnames_models:
            type: dict
            meaning: keys are model types (i.e. 'VARIMA', 'LightGBMModel', etc), values are model objects
        df:
            type: pd.DataFrame
            meaning: data to run through model for preds
        time_col:
            type: str
            meaning: name of timestamp column
        forecast_horizon:
            type: int
            meaning: number of steps into the future to predict
        LAG_MIN:
            type: int
            meaning: minimum lag value (default=3)
    Outputs:
        modnames_preds:
            type: dict
            meaning: keys are model types, values are pred values for all features (as dfs)
    """

    print('Getting modnames_preds...')
    modnames_preds = {}
    # Get preds -- rolling forward in time
    for mod_name, model in modnames_models.items():
        features = list(model.training_series.components)
        preds = get_preds_rolling(model=model,
                                    df=df,
                                    features=features,
                                    LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
                                    time_col=time_col,
                                    forecast_horizon=forecast_horizon)
        df_preds = pd.DataFrame(preds, columns=features)
        time_vals = df[time_col].values[-df_preds.shape[0]:]
        df_preds.insert(0, time_col, time_vals)
        modnames_preds[mod_name] = df_preds
    print('  --> done')
    return modnames_preds


def plot_predtrue(pred, true, mod_name, time_col, output_dir):
    """
    Purpose:
        Plot pred vs true values for each model type and pred feature
    Inputs:
        pred:
            type: pd.DataFrame
            meaning: pred values for all pred features
        true:
            type: pd.DataFrame
            meaning: true values for all pred features
        mod_name:
            type: str
            meaning: name of model type (i.e. 'VARIMA', 'LightGBMModel', etc)
        output_dir:
            type: str
            meaning: dir to save plot too
    Outputs:
        n/a (plots saved)
    """
    xs = [_ for _ in range(pred.shape[0])]
    for pcol in pred:
        if pcol == time_col:
            continue
        plt.cla()
        path_out = os.path.join(output_dir, f"{mod_name}--{pcol}")
        plt.plot(xs, pred[pcol], label='pred')
        plt.plot(xs, true[pcol], label='true')
        plt.xlabel('time step')
        plt.ylabel(f'{pcol}')
        plt.title(f"Pred vs True -- {pcol}")
        plt.legend()
        plt.savefig(path_out)


def get_modnames_evals(modnames_preds, true, time_col, eval_metric, output_dir):
    """
    Purpose:
        Get eval metric scores for all model types
    Inputs:
        modnames_preds:
            type: dict
            meaning: keys are model types, values are pred values for all features (as dfs)
        true:
            type: pd.DataFrame
            meaning: true values for all pred features
        time_col:
            type: str
            meaning: name of timestamp column
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
        output_dir:
            type: str
            meaning: dir to save plot too
    Outputs:
        modnames_evals:
            type: dict
            meaning: keys are model types, values are eval metric scores
    """
    print('Getting modnames_evals...')
    modnames_evals = {}
    for mod_name, preds in modnames_preds.items():
        ts_pred = TimeSeries.from_dataframe(preds, time_col=time_col)
        ts_true = TimeSeries.from_dataframe(true.tail(len(preds)), time_col=time_col)
        # Eval pred vs true
        modnames_evals[mod_name] = get_eval(ts_pred, ts_true, time_col, eval_metric)
        # Plot pred vs true
        plot_predtrue(preds, true.tail(len(preds)), mod_name, time_col, output_dir)
    print('  --> done')
    return modnames_evals


def gridsearch_model(model, mod_name, mod_grid, data_t0, forecast_horizon, eval_metric, time_col=None, verbose=True):
    """
    Purpose:
        Find best params for model type given 'mod_grid'
    Inputs:
        model:
            type: darts.ForecastingModel
            meaning: model object to run gridsearch on
        mod_name:
            type: str
            meaning: name of model type (i.e. 'VARIMA', 'LightGBMModel', etc)
        mod_grid:
            type: dict
            meaning: keys are model types, values are param dicts for gridsearch
        data_t0:
            type: darts.TimeSeries
            meaning: data to use for gridseach
        forecast_horizon:
            type: int
            meaning: number of steps into the future to predict
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
        time_col:
            type: str
            meaning: name of timestamp column
        verbose:
            type: bool
            meaning: whether to active verbose in gridsearch`
    Outputs:
        model_best_untrained:
            type: darts.ForecastingModel
            meaning: untrained model object with best_params
        best_params:
            type: dict
            meaning: best params for 'model' from 'mod_grid'
        score:
            type: float
            meaning: eval score for best model
    """
    model_best_untrained, best_params, best_score = model.gridsearch(parameters=mod_grid,
                                                                        series=data_t0,
                                                                        forecast_horizon=forecast_horizon,
                                                                        verbose=verbose,
                                                                        metric=METRICNAMES_METRICS[eval_metric])
    return model_best_untrained, best_params, best_score


def get_model_best(modnames_scores, eval_metric):
    """
    Purpose:
        Get best performing model type
    Inputs:
        modnames_scores:
            type: dict
            meaning: keys are model types, values are eval metric scores
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
    Outputs:
        best_mod:
            type: str
            meaning: model type with best eval metric score
    """
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


def save_results(modnames_params: dict, modnames_scores: dict, output_dir: str, filename: str, eval_metric:str):
    """
    Purpose:
        Save best params (from gridsearch) and eval scores for each model type
    Inputs:
        modnames_params:
            type: dict
            meaning: keys are model types, values are best params (from gridsearch)
        modnames_scores:
            type: dict
            meaning: keys are model types, values are eval metric scores
        output_dir:
            type: str
            meaning: dir to save plot too
        filename:
            type: str
            meaning: name of csv file saved
        eval_metric:
            type: str
            meaning: name of function to eval model preds vs true (i.e. 'rmse', 'mape', etc)
    Outputs:
        n/a (csvs saved)
    """
    results = {'name': [], 'params':[], 'eval':[]}
    for mod_name, mod_params in modnames_params.items():
        results['name'].append(mod_name)
        results['params'].append(mod_params)
        results['eval'].append(modnames_scores[mod_name])
    ascending = True
    if EVALS_COMPARES[eval_metric] == operator.gt:
        ascending = False
    results = pd.DataFrame(results).sort_values(by='eval', ascending=ascending)
    path_out = os.path.join(output_dir, f'{filename}.csv')
    results.to_csv(path_out, index=False)


def get_modname(model):
    """
    Purpose:
        Get name of model type from model
    Inputs:
        model:
            type: darts.models obj
            meaning: ML model
    Outputs:
        mod_name:
            type: str
            meaning: name of model type (i.e. 'VARIMA', 'LightGBMModel')
    """
    mod_name = None
    for modname, mod in MODNAMES_MODELS.items():
        if isinstance(model, mod):
            mod_name = modname
    return mod_name