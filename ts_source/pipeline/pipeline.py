import os
import sys
import pandas as pd
from typing import List, Optional, Sequence, Tuple, Union

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_save_models, get_model_best, get_modnames_preds, get_modnames_evals, save_results
from ts_source.preprocess.preprocess import split_data, check_stationarity, scale_data
from ts_source.utils.utils import get_args, load_config, validate_config, validate_args, save_data, load_models, make_dir


def run_pipeline(
            config:dict,
            data_path:Union[bool,str]=False,
            output_dir:Union[bool,str]=False,
            data:Union[bool,pd.DataFrame]=False,
            output_dirs:Union[bool,dict]=False,
            scaler:Optional[bool]=False
            ):
    """
    Purpose:
        Run pipeline, training+inference or just inference
    Inputs:
        config:
            type: dict
            meaning: yaml dict with configuration for pipeline run
        data:
            type: bool or pd.DataFrame
            meaning: data to run if df, if bool data will be loaded from 'data_path' arg
        data_path:
            type: bool or str
            meaning: path to load data from if str, if bool data comes from 'data' arg
        output_dir:
            type: bool or str
            meaning: dir to save outputs too if str, if bool out dirs come from 'output_dirs' arg
        output_dirs:
            type: dict or str
            meaning: dict with output sub-dirs ('data', 'models' & 'results'), if bool out dirs made within 'output_dir' arg
    Outputs:
        modnames_models:
            type: dict
            meaning: keys are model types (i.e. 'VARIMA', 'LightGBMModel', etc), values are model objects
        modname_best:
            type: str
            meaning: highest performing model type
        modnames_preds:
            type: dict
            meaning: keys are model types, values are pred values for all features (as dfs)
    """
    data                                                            = validate_args(config, data_path, output_dir, data, output_dirs)
    config                                                          = validate_config(config, data, output_dir, output_dirs)
    if config['scale']:
        data, scaler                                                = scale_data(data, config['features']['in'], config['scale'], rescale=False)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'], config['train_models'])
    save_data(data_dict, config['dirs']['data'])
    check_stationarity(data_dict['t0t1'], config['time_col'], config['dirs']['data'])
    if config['train_models']: # training mode, infer on test_prop%
        modnames_models, modnames_params, modnames_evals_train      = train_save_models(data_dict, config['do_gridsearch'], config['modnames_grids'], config['dirs']['models'], config['time_col'], config['eval_metric'], config['forecast_horizon'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t1'], config['time_col'], config['forecast_horizon'], scaler)
        modnames_evals_test                                         = get_modnames_evals(modnames_preds, data_dict['t1'], config['time_col'], config['eval_metric'], config['dirs']['results'])
        save_results(modnames_params, modnames_evals_train, config['dirs']['results'], 'best_params--train', config['eval_metric'])
        save_results(modnames_params, modnames_evals_test, config['dirs']['results'], 'best_params--test', config['eval_metric'])
    else: # inference mode, infer on 100%
        modnames_models                                             = load_models(config['dirs']['models'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t0t1'], config['time_col'], config['forecast_horizon'], scaler)
        modnames_evals_test                                         = get_modnames_evals(modnames_preds, data_dict['t0t1'], config['time_col'], config['eval_metric'], config['dirs']['results'])
    modname_best                                                    = get_model_best(modnames_evals_test, config['eval_metric'])
    save_data(modnames_preds, config['dirs']['results'])
    return modnames_models, modname_best, modnames_preds


if __name__ == '__main__':
    config                                              = load_config(get_args().config_path)
    modnames_models, modname_best, modnames_preds       = run_pipeline(config,
                                                                        data_path=get_args().data_path,
                                                                        output_dir=get_args().output_dir,
                                                                        data=False,
                                                                        output_dirs=False)
    print('\n  DONE')
