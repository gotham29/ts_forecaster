import os
import sys
import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.model.model import train_models, test_models, get_model_best
from ts_source.preprocess.preprocess import split_data
from ts_source.utils.utils import get_args, load_config, validate_config, save_data, load_models, save_models, save_data_as_pickle, load_pickle_object_as_data


def save_results(modnames_params: dict, modnames_scores: dict, dir_out: str, name: str):
    results = {'name': [], 'params':[], 'score':[]}
    for mod_name, mod_params in modnames_params.items():
        results['name'].append(mod_name)
        results['params'].append(mod_params)
        results['score'].append(modnames_scores[mod_name])
    results = pd.DataFrame(results).sort_values(by='score', ascending=True)
    path_out = os.path.join(dir_out, f'{name}.csv')
    results.to_csv(path_out, index=False)


def run_pipeline(config, data=False, modnames_testinfs={}):
    if isinstance(data, bool):
        data                                                        = pd.read_csv(config['dirs']['data_in'])
    config                                                          = validate_config(config, data)
    data_dict                                                       = split_data(data, config['data_cap'], config['time_col'], config['features'], config['test_prop'])
    save_data(data_dict, config['dirs']['data_out'])
    if config['train_models']: # training mode, test on test_prop%
        modnames_models, modnames_params, modnames_trainlosses      = train_models(data_dict, config['modnames_grids'], config)
        save_models(modnames_models, config['dirs']['models_out'])
        save_results(modnames_params, modnames_trainlosses, config['dirs']['results_out'], 'train')
        save_data_as_pickle(modnames_params, os.path.join(config['dirs']['results_out'], 'best_params.pkl') )
        modnames_testlosses                                         = test_models(modnames_models, data_dict['t1'], config['time_col'], config['loss_metric'])
        save_results(modnames_params, modnames_testlosses, config['dirs']['results_out'], 'test')
    else: # inference mode, test on 100%
        modnames_models                                             = load_models(config['dirs']['models_out'])
        modnames_params                                             = load_pickle_object_as_data(os.path.join(config['dirs']['results_out'], 'best_params.pkl') )
        modnames_testinfs = get_preds(modnames_models, data_dict['t0t1'], config['time_col'])
        # modnames_testlosses                                         = test_models(modnames_models, data_dict['t0t1'], config['time_col'], config['loss_metric'] )
    modname_best                                                    = get_model_best(modnames_testlosses)
    return modnames_models, modname_best, modnames_testinfs


if __name__ == '__main__':
    config                          = load_config(get_args().config_path)
    modnames_models, modname_best   = run_pipeline(config)
    print('\n  DONE')

""" TEST """
# config_path = "/Users/samheiserman/Desktop/PhD/Motion-Print/configs/run_pipeline.yaml"
# config = load_config(config_path)
# run_pipeline(config)
