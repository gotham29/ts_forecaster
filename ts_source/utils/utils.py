import argparse
import os
import pickle
import pandas as pd
import yaml


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cp', '--config_path', required=True,
                        help='path to config')
    return parser.parse_args()


def make_dir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def get_dirfiles(dir_root, files_types=None):
    subfiles = []
    for path, subdirs, files in os.walk(dir_root):
        for name in files:
            subfiles.append(os.path.join(path, name))
    if files_types is not None:
        assert isinstance(files_types, list), f"files_types should be a list, found --> {type(files_types)}"
        subfiles = [s for s in subfiles if s.split('.')[-1] in files_types]
    return subfiles


def load_config(yaml_path):
    """
    Purpose:
        Load config from path
    Inputs:
        yaml_path
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_data_as_pickle(data_struct, f_path):
    """
    Saves a dictionary as pickle object to f_path
    :param data_struct: Data object (dict/list) you want to save
    :param f_path: File path
    :return: True flag
    """
    with open(f_path, 'wb') as handle:
        pickle.dump(data_struct, handle)
    return True


def validate_config(config, data):
    print('\nValidating Config...')
    # Ensure expected keys present and correct type
    keys_dtypes = {
        'test_prop': float,
        'dirs': dict,
        'features': dict,
        'modnames_grids': dict,
        'data_cap': int,
        'forecast_horizon': int,
        'loss_metric': str,
        'time_col': str
    }
    keys_missing = []
    keys_wrongdtypes = {}
    for k, dtype in keys_dtypes.items():
        if k not in config:
            keys_missing.append(k)
        continue
        if not isinstance(config[k], dtype):
            keys_wrongdtypes[k] = type(config[k])
    assert len(keys_missing) == 0, f"  Expected keys missing --> {sorted(keys_missing)}"
    for k, wrongdtype in keys_wrongdtypes.items():
        print(f"  {k} found type -->{wrongdtype}; expected --> {keys_dtypes[k]}")
    assert len(keys_wrongdtypes) == 0, "  wrong data types"

    # Ensure paths exist
    assert os.path.exists(config['dirs']['data_in']), f"data path not found!\n  --> {config['dirs']['data_in']}"
    make_dir(config['dirs']['models_out'])
    make_dir(config['dirs']['results_out'])

    # Ensure test prop between 0.1 and 0.7
    assert 0.1 <= config['test_prop'] <= 0.7, f"test_prop expected between 0.1 and 0.7! found\n  --> {config['test_prop']}"

    # Ensure data_cap between 300 and 100000
    assert 300 <= config['data_cap'] <= 100000, f"data_cap expected between 100 and 100000! found\n  --> {config['data_cap']}"

    known_metrics = ['mape', 'mase']
    # Ensure loss_metric is known
    assert config['loss_metric'] in known_metrics, f"loss_metric unknown! --> {config['loss_metric']}\nKnown --> {known_metrics}"

    # Ensure time_col in data
    assert config['time_col'] in data, f"time_col not found in data! --> {config['time_col']}\nFound --> {sorted(data.columns)}"

    # Ensure forecast_horizon between 1 and 100
    assert 1 <= config['forecast_horizon'] <= 100, f"forecast_horizon expected between 1 and 100! found\n  --> {config['forecast_horizon']}"

    # Ensure all feaurtes in data
    features_missing = []
    feat_types = ['in', 'pred']
    for feat_type, features in config['features'].items():
        assert feat_type in feat_types, f"features lists should be in {feat_types}! found\n  --> {feat_type}"
        assert isinstance(features, list), f"features[{feat_type}] should be list! found\n  --> {type(features)}"
        for feat in features:
            if feat not in data:
                features_missing.append(feat)
    assert len(features_missing) == 0, f"features missing!\n  --> {features_missing}"

    # Ensure all feaurtes numeric
    features_data = {}
    features_nonnumeric = []
    for feat_type, features in config['features'].items():
            for feat in features:
                try:
                    features_data[feat] = data[feat].astype(float)
                except:
                    features_nonnumeric.append(feat)
    assert len(features_nonnumeric) == 0, f"non-numeric features found!\n  --> {features_nonnumeric}"

    # Ensure all features have at least 3 unique values in first 1000
    min_unique = 3
    features_nonunique = []
    for feat, feat_data in features_data.items():
        feat_uni = len(feat_data[:1000].unique())
        if feat_uni < min_unique:
            features_nonunique.append(feat)
    assert len(features_nonunique) == 0, f"features with < {min_unique} unique values found!\n  --> {features_nonunique}"

    # Ensure modnames are known
    modnames_known = ['VARIMA', 'NBEATSModel', 'TCNModel',
                        'TransformerModel', 'RNNModel', 'LightGBMModel']
    modnames_unknown = []
    for modname in config['modnames_grids']:
        if modname not in modnames_known:
            modnames_unknown.append(modname)
    assert len(modnames_unknown) == 0, f"unknown models found!\n  --> {modnames_unknown}"

    # Add missing keys from modgrids
    total_length = data[:config['data_cap']].shape[0]
    train_length = int(total_length*(1-config['test_prop']))
    test_length = int(total_length*config['test_prop'])
    ## NBEATS
    modnames_missingvals = {
        'NBEATSModel': {
            'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
            },
        'TCNModel': {
            'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
            },
        'TransformerModel': {
            'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
            },
        'RNNModel': {
            'training_length': [ train_length ]
            },
        'LightGBMModel': {
            'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
            },
    }
    for mod_name, missingvals in modnames_missingvals.items():
        for vname, v in missingvals.items():
            config['modnames_grids'][f'{mod_name}'][f'{vname}'] = v

    return config


def load_models(dir_models):
    """
    Purpose:
        Load pkl models for each feature from dir
    Inputs:
        dir_models
            type: str
            meaning: path to dir where pkl models are loaded from
    Outputs:
        features_models
            type: dict
            meaning: model obj for each feature
    """
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    print(f"Loading {len(pkl_files)} models...")
    features_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        features_models[f.replace('.pkl', '')] = model
    return features_models


def save_models(modnames_models, dir_out):
    print(f'Saving {len(modnames_models)} models...')
    for modname, model in modnames_models.items():
        path_out = os.path.join(dir_out, f"{modname}.pkl")
        save_data_as_pickle(model, path_out)


def load_pickle_object_as_data(file_path):
    """
    Purpose:
        Load data from pkl file
    Inputs:
        file_path
            type: str
            meaning: path to pkl file
    Outputs:
        data
            type: pkl
            meaning: pkl data loaded
    """
    with open(file_path, 'rb') as f_handle:
        data = pickle.load(f_handle)
    return data


def save_data(data_dict, dir_out):
    print(f'Saving prepped data...\n  to --> {dir_out}')
    for dname, data in data_dict.items():
        path_out = os.path.join(dir_out, f"{dname}.csv")
        data.to_csv(path_out)


