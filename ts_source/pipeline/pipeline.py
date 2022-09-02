import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_models, get_model_best, get_modnames_preds, get_loss, get_modnames_losses, save_results  #, test_models
from ts_source.preprocess.preprocess import split_data
from ts_source.utils.utils import get_args, load_config, validate_config, save_data, load_models, save_models


def run_pipeline(config, data=False, modname_best=None, modnames_preds={}):
    if isinstance(data, bool):
        data                                                        = pd.read_csv(config['dirs']['data_in'])
    config                                                          = validate_config(config, data)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'])
    save_data(data_dict, config['dirs']['data_out'])
    if config['train_models']: # training mode, test on test_prop%
        modnames_models, modnames_params, modnames_losses_train     = train_models(data_dict, config['modnames_grids'], config)
        save_models(modnames_models, config['dirs']['models_out'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t1'], config['time_col'], config['forecast_horizon'])
        modnames_losses_test                                        = get_modnames_losses(modnames_preds, data_dict['t1'], config['time_col'], config['loss_metric'])
        modname_best                                                = get_model_best(modnames_losses)
        save_results(modnames_params, modnames_losses_train, config['dirs']['results_out'], 'train')
        save_results(modnames_params, modnames_losses_test, config['dirs']['results_out'], 'test')
    else: # inference mode, test on 100%
        modnames_models                                             = load_models(config['dirs']['models_out'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t0t1'], config['time_col'], config['forecast_horizon'])
    return modnames_models, modname_best, modnames_preds


if __name__ == '__main__':
    config                                              = load_config(get_args().config_path)
    modnames_models, modname_best, modnames_testinfs    = run_pipeline(config)
    print('\n  DONE')

""" TEST """
# config_path = "/Users/samheiserman/Desktop/PhD/Motion-Print/configs/run_pipeline.yaml"
# config = load_config(config_path)
# run_pipeline(config)
