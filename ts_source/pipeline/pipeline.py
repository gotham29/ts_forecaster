import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_save_models, get_model_best, get_modnames_preds, get_modnames_evals, save_results
from ts_source.preprocess.preprocess import split_data
from ts_source.utils.utils import get_args, load_config, validate_config, save_data, load_models #, save_models


def run_pipeline(config, data=False, modname_best=None):
    if isinstance(data, bool):
        data                                                        = pd.read_csv(config['dirs']['data_in'])
    config                                                          = validate_config(config, data)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'])
    save_data(data_dict, config['dirs']['data_out'])
    if config['train_models']: # training mode, test on test_prop%
        modnames_models, modnames_params, modnames_evals_train      = train_save_models(data_dict, config['modnames_grids'], config['dirs']['models_out'], config['time_col'], config['eval_metric'], config['forecast_horizon'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t1'], config['time_col'], config['forecast_horizon'])
        modnames_evals_test                                         = get_modnames_evals(modnames_preds, data_dict['t1'], config['time_col'], config['eval_metric'])
        modname_best                                                = get_model_best(modnames_evals_test, config['eval_metric'])
        save_results(modnames_params, modnames_evals_train, config['dirs']['results_out'], 'train')
        save_results(modnames_params, modnames_evals_test, config['dirs']['results_out'], 'test')
    else: # inference mode, test on 100%
        modnames_models                                             = load_models(config['dirs']['models_out'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t0t1'], config['time_col'], config['forecast_horizon'])
    save_data({'modnames_preds': pd.DataFrame(modnames_preds)}, config['dirs']['results_out'])
    return modnames_models, modname_best, modnames_preds


if __name__ == '__main__':
    config                                              = load_config(get_args().config_path)
    modnames_models, modname_best, modnames_preds       = run_pipeline(config)
    print('\n  DONE')

