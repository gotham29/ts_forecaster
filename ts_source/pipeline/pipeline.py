import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_save_models, get_model_best, get_modnames_preds, get_modnames_evals, save_results
from ts_source.preprocess.preprocess import split_data
from ts_source.utils.utils import get_args, load_config, validate_config, save_data, load_models, make_dir


def validate_args(config, data_path, output_dir, data, output_dirs):

    # Check which args are found
    found_data = (not isinstance(data, bool))
    found_datapath = (not isinstance(data_path, bool))
    found_outputdir = (not isinstance(output_dir, bool))
    found_outputdirs = (output_dirs != {})

    # Ensure 1 of 'data_path' or 'data' is found
    assert sum([found_data, found_datapath]) == 1, "just 1 of 'data' or 'data_path' should be passed"
    if found_datapath:
        # Ensure 'data_path' exists and is csv
        assert os.path.exists(data_path), f"'data_path' not found!\n --> {data_path}"
        file_type = data_path.split('.')[-1]
        assert file_type == 'csv', f"'data_path' expected .csv\n  found --> {file_type}"
        data = pd.read_csv(data_path)
    else:
        # Ensure 'data' is pd.DataFrame
        assert isinstance(data, pd.DataFrame), f"'data' expected pd.DataFrame\n  found --> {type(data)}"

    # Ensure 1 of 'output_dir' or 'output_dirs' is found
    assert sum([found_outputdir, found_outputdirs]) == 1, "just 1 of 'output_dir' or 'output_dirs' should be passed"
    if found_outputdir:
        # Ensure output_dir exists
        make_dir(output_dir)
    else:
        # Ensure dir_names are valid and dirs exist
        dirnames_valid, dirnames_invalid = ['data', 'models', 'results', 'scalers'],[]
        for dir_name, dir_ in output_dirs.items():
            if dir_name not in dirnames_valid:
                dirnames_invalid.append(dir_name)
            make_dir(dir_)
        assert len(dirnames_invalid) == 0, f"invalid dir_names found in 'output_dirs'!\n  found --> {dirnames_invalid}\n  valid --> {dirnames_valid}"

    return data


def run_pipeline(config: dict, data_path=False, output_dir=False, data=False, output_dirs=False, modname_best=None):
    data                                                            = validate_args(config, data_path, output_dir, data, output_dirs)
    config                                                          = validate_config(config, data, output_dir, output_dirs)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'], config['train_models'])
    save_data(data_dict, config['dirs']['data'])
    if config['train_models']: # training mode, infer on test_prop%
        modnames_models, modnames_params, modnames_evals_train      = train_save_models(data_dict, config['modnames_grids'], config['dirs']['models'], config['time_col'], config['eval_metric'], config['forecast_horizon'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t1'], config['time_col'], config['forecast_horizon'])
        modnames_evals_test                                         = get_modnames_evals(modnames_preds, data_dict['t1'], config['time_col'], config['eval_metric'], config['dirs']['results'])
        modname_best                                                = get_model_best(modnames_evals_test, config['eval_metric'])
        save_results(modnames_params, modnames_evals_train, config['dirs']['results'], 'best_params--train', config['eval_metric'])
        save_results(modnames_params, modnames_evals_test, config['dirs']['results'], 'best_params--test', config['eval_metric'])
    else: # inference mode, infer on 100%
        modnames_models                                             = load_models(config['dirs']['models'])
        modnames_preds                                              = get_modnames_preds(modnames_models, data_dict['t0t1'], config['time_col'], config['forecast_horizon'])
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
