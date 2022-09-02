import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_save_models, get_model_best, get_modnames_preds, get_modnames_evals, save_results
from ts_source.preprocess.preprocess import split_data
from ts_source.utils.utils import get_args, load_config, validate_config, save_data, load_models #, save_models


def run_pipeline(config: dict, output_dir: str, data=False, data_path=False, modname_best=None):
    assert not (data==False and data==False), f"run_pipeline needs either 'data'(pd.DataFrame) or 'data_path'(csv)"
    if data==False:
        data                                                        = pd.read_csv(data_path)
    config                                                          = validate_config(config, data, output_dir)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'])
    save_data(data_dict, config['dirs']['data'])
    if config['train_models']: # training mode, infer on test_prop%
        modnames_models, modnames_params, modnames_evals_train      = train_save_models(data_dict, config['modnames_grids'], config['dirs']['models'], config['time_col'], config['eval_metric'], config['forecast_horizon'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t1'], config['time_col'], config['forecast_horizon'])
        modnames_evals_test                                         = get_modnames_evals(modnames_preds, data_dict['t1'], config['time_col'], config['eval_metric'])
        modname_best                                                = get_model_best(modnames_evals_test, config['eval_metric'])
        save_results(modnames_params, modnames_evals_train, config['dirs']['results'], 'train', config['eval_metric'])
        save_results(modnames_params, modnames_evals_test, config['dirs']['results'], 'test', config['eval_metric'])
    else: # inference mode, infer on 100%
        modnames_models                                             = load_models(config['dirs']['models'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t0t1'], config['time_col'], config['forecast_horizon'])
    save_data({'modnames_preds': pd.DataFrame(modnames_preds)}, config['dirs']['results'])
    return modnames_models, modname_best, modnames_preds


if __name__ == '__main__':
    config                                              = load_config(get_args().config_path)
    modnames_models, modname_best, modnames_preds       = run_pipeline(config, data_path=get_args().data_path, output_dir=get_args().output_dir)
    print('\n  DONE')

