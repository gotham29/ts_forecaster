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
                                                            loss_metric=config['loss_metric'],
                                                            forecast_horizon=config['forecast_horizon'],
                                                            time_col=config['time_col'],
                                                            verbose=False)
        modnames_models[mod_name] = model_untrained.fit(series=darts_timeseries)
        modnames_params[mod_name] = params
        modnames_scores[mod_name] = score
        print(f"    best params --> {params}")
    return modnames_models, modnames_params, modnames_scores


def get_loss(preds, df_test, time_col, loss_metric):
    actuals = TimeSeries.from_dataframe(df_test, time_col=time_col)
    loss = METRICNAMES_METRICS[loss_metric](actual_series=actuals, pred_series=preds)
    return loss


def get_modnames_preds(modnames_models, df, time_col, forecast_horizon):
    modnames_preds = {}
    series = TimeSeries.from_dataframe(df, time_col=time_col)
    for mod_name, model in modnames_models.items():
        modnames_preds[mod_name] = model.predict(n=forecast_horizon, series=[series], past_covariates=None, future_covariates=None) #.predict(len(df_test))


def get_modnames_losses(modnames_preds, df, time_col, loss_metric):
    modnames_losses = {}
    for mod_name, preds in modnames_preds.items():
        modnames_losses[mod_name] = get_loss(preds, df, time_col, loss_metric)
    return modnames_losses


# def test_models(modnames_models, df_test, time_col, loss_metric, forecast_horizon):
#     print(f"\nTesting {len(modnames_models)} models on df_test:{df_test.shape}...")
#     # Get models predictions

#     # modnames_preds = {}
#     # series = TimeSeries.from_dataframe(df_test, time_col=time_col)
#     # for mod_name, model in modnames_models.items():
#     #     modnames_preds[mod_name] = model.predict(n=forecast_horizon, series=[series], past_covariates=None, future_covariates=None) #.predict(len(df_test))
#     modnames_preds = get_modnames_preds(modnames_models, df_test, time_col, forecast_horizon)

#     # Scores predictions vs true
#     # modnames_losses = {}
#     # for mod_name, preds in modnames_preds.items():
#     #     modnames_losses[mod_name] = get_loss(preds, df_test, time_col, loss_metric)
#     modnames_losses = get_modnames_losses(modnames_preds, df_test, time_col, loss_metric)
#     return modnames_losses


def gridsearch_model(model, mod_name, mod_grid, data_t0, forecast_horizon, loss_metric, time_col=None, verbose=True):
    model_best = model.gridsearch(parameters=mod_grid,
                                    series=data_t0,  #TimeSeries.from_dataframe(df=data_t0, time_col=time_col),
                                    forecast_horizon=forecast_horizon,
                                    verbose=verbose,
                                    metric=METRICNAMES_METRICS[loss_metric])
    return model_best


def get_model_best(modnames_scores):
    print("Getting best model...")
    best_loss, best_mod = np.inf, None
    for mod_name, loss_metric in modnames_scores.items():
        if loss_metric < best_loss:
            best_loss = loss_metric
            best_mod = mod_name
            print(f"  {best_loss} <-- {best_mod}")
    print(f"    *{best_mod}*")
    return best_mod

