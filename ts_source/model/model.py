import sys
import time
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

EVALS_BETTER = {
    'mae':'lower',
    'mse':'lower',
    'rmse':'lower',
    'mape':'lower',
    'mase':'lower',
    'ope':'lower',
    'marre':'lower',
    'dtw_metric':'lower',
    'r2_score':'higher',
}


def train_models(data_dict:dict, modnames_grids: dict, config: dict):
    print(f'Training {len(modnames_grids)} models...')
    # Convert df to darts timeseries
    darts_timeseries = TimeSeries.from_dataframe(df=data_dict['t0'], time_col=config['time_col'])
    # Train all models
    modnames_models, modnames_params, modnames_scores = {}, {}, {}
    for mod_name, mod_grid in modnames_grids.items():
        print(f"  mod_name = {mod_name}")
        model_untrained, params, score = gridsearch_model(model=MODNAMES_MODELS[mod_name],
                                                            mod_name=mod_name,
                                                            mod_grid=mod_grid,
                                                            data_t0=darts_timeseries,
                                                            eval_metric=config['eval_metric'],
                                                            forecast_horizon=config['forecast_horizon'],
                                                            time_col=config['time_col'],
                                                            verbose=False)
        modnames_models[mod_name] = model_untrained.fit(series=darts_timeseries)
        modnames_params[mod_name] = params
        modnames_scores[mod_name] = score
        print(f"    best params --> {params}")
    return modnames_models, modnames_params, modnames_scores


def get_eval(preds, df, time_col, eval_metric):
    actuals = TimeSeries.from_dataframe(df, time_col=time_col)
    eval_metric = METRICNAMES_METRICS[eval_metric](actual_series=actuals, pred_series=preds)
    return eval_metric


def get_model_lag(mod_name, model):
    lag_param = MODNAMES_LAGPARAMS[mod_name]
    return model.model_params[lag_param]


def get_modnames_preds(modnames_models, df, time_col, forecast_horizon, LAG_MIN=3):
    print('Getting modnames_preds...')
    modnames_preds = {}
    # Get rolling preds
    for mod_name, model in modnames_models.items():
        LAG = max(LAG_MIN, get_model_lag(mod_name, model))
        n_features = len(model.training_series.components)
        preds = []
        for _ in range(df.shape[0]):
            if _ < LAG:
                continue
            df_lag = df[_-LAG:_]
            ts = TimeSeries.from_dataframe(df_lag, time_col=time_col) #df_row
            pred = model.predict(n=forecast_horizon, series=ts)
            preds.append(pred.data_array().values.reshape(n_features))
        modnames_preds[mod_name] = preds
    print('  --> done')
    return modnames_preds


def get_modnames_evals(modnames_preds, df, time_col, eval_metric):
    print('Getting modnames_evals...')
    modnames_evals = {}
    for mod_name, preds in modnames_preds.items():
        modnames_evals[mod_name] = get_eval(preds, df, time_col, eval_metric)
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
    # Check how to use eval metric
    better = EVALS_BETTER[eval_metric]
    print(f"  {eval_metric} better = {better}")
    comp = operator.lt
    if better == 'higher':
        comp = operator.gt
    # Get best model
    best_metric, best_mod = np.inf, None
    for mod_name, current_metric in modnames_scores.items():
        if comp(current_metric, best_metric): #current_metric vs best_metric:
            best_metric = current_metric
            best_mod = mod_name
            print(f"  {best_metric} <-- {best_mod}")
    print(f"    *{best_mod}*")
    return best_mod


def save_results(modnames_params: dict, modnames_scores: dict, dir_out: str, name: str):
    results = {'name': [], 'params':[], 'score':[]}
    for mod_name, mod_params in modnames_params.items():
        results['name'].append(mod_name)
        results['params'].append(mod_params)
        results['score'].append(modnames_scores[mod_name])
    results = pd.DataFrame(results).sort_values(by='score', ascending=True)
    path_out = os.path.join(dir_out, f'{name}.csv')
    results.to_csv(path_out, index=False)