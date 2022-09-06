import argparse
import os
import pickle
import pandas as pd
import yaml
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel


MODNAMES_KNOWN = ['VARIMA', 'NBEATSModel', 'TCNModel',
                    'TransformerModel', 'RNNModel', 'LightGBMModel',
                    'KalmanForecaster']


def get_args():
    """
    Purpose:
        Get args from command line call
    Inputs:
        -cp:
            type: str
            meaning: path to config (yaml)
        -dp:
            type: str
            meaning: path to data path (csv)
        -od:
            type: str
            meaning: dir to save outputs to
    Outputs:
        parser.parse_args()
            type: parser
            meaning: container for -cp, -dp and -od
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cp', '--config_path', required=True,
                        help='path to config')
    parser.add_argument('-dp', '--data_path', required=True,
                        help='path to data')
    parser.add_argument('-od', '--output_dir', required=True,
                        help='dir to send outputs')
    return parser.parse_args()


def make_dir(mydir):
    """
    Purpose:
        Make dir if doesn't already exist
    Inputs:
        mydir:
            type: str
            meaning: dir to make
    Outputs:
        n/a (dir made)
    """
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def get_dirfiles(dir_root, files_types=None):
    """
    Purpose:
        Get all paths within 'dir_root' of type in 'files_types'
    Inputs:
        dir_root:
            type: str
            meaning: dir to find paths in
        files_types:
            type: list
            meaning: file types to find in 'dir_root'
    Outputs:
        subfiles:
            type: list
            meaning: list of paths in 'dir_root' of type in 'files_types'
    """
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
        yaml_path:
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg:
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_data_as_pickle(data_struct, f_path):
    """
    Purpose:
        Save data in pkl format
    Inputs:
        data_struct:
            type: anythin pickle-able (dict/list)
            meaning: data to be saved as pkl
        f_path:
            type: str
            meaning: path to save data to
    Outputs:
        n/a (pkl data saved)
    """
    with open(f_path, 'wb') as handle:
        pickle.dump(data_struct, handle)


def validate_config(config, data, output_dir, output_dirs):
    """
    Purpose:
        Ensure validity of all config values
    Inputs:
        config:
            type: dict
            meaning: yaml dict with configuration for pipeline run to validate
        data:
            type: bool or pd.DataFrame
            meaning: data to run if df, if bool data will be loaded from 'data_path' arg
        output_dir:
            type: bool or str
            meaning: dir to save outputs too if str, if bool out dirs come from 'output_dirs' arg
        output_dirs:
            type: dict or str
            meaning: dict with output sub-dirs ('data', 'models' & 'results'), if bool out dirs made within 'output_dir' arg
    Outputs:
        config:
            type: dict
            meaning: validated yaml dict
    """
    print('\nValidating Config...')
    # Ensure expected keys present and correct type
    keys_dtypes = {
        'test_prop': float,
        'features': dict,
        'modnames_grids': dict,
        'data_cap': int,
        'forecast_horizon': int,
        'eval_metric': str,
        'time_col': str,
        'train_models': bool
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
    if output_dir:
        outdirs = ['data', 'models', 'results']
        config['dirs'] = {}
        make_dir(output_dir)
        for od in outdirs:
            od_path = os.path.join(output_dir, od)
            config['dirs'][od] = od_path
            make_dir(od_path)
    else:
        config['dirs'] = output_dirs

    # Ensure test prop between 0.1 and 0.7
    assert 0.1 <= config['test_prop'] <= 0.7, f"test_prop expected between 0.1 and 0.7! found\n  --> {config['test_prop']}"

    # Ensure data_cap between 300 and 100000
    assert 50 <= config['data_cap'] <= 100000, f"data_cap expected between 50 and 100000! found\n  --> {config['data_cap']}"

    known_metrics = ['mae', 'mse', 'rmse', 'mape', 'mase', 'ope', 'marre', 'r2_score', 'dtw_metric']
    # Ensure eval_metric is known
    assert config['eval_metric'] in known_metrics, f"eval_metric unknown! --> {config['eval_metric']}\nKnown --> {known_metrics}"

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
    modnames_unknown = []
    for modname in config['modnames_grids']:
        if modname not in MODNAMES_KNOWN:
            modnames_unknown.append(modname)
    assert len(modnames_unknown) == 0, f"unknown models found!\n  --> {modnames_unknown}\n  known --> {MODNAMES_KNOWN}"

    # Add missing keys to modnames_grids
    total_length = data[:config['data_cap']].shape[0]
    train_length = int(total_length*(1-config['test_prop']))
    test_length = int(total_length*config['test_prop'])
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
        'LightGBMModel': {
            'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
            },
        # 'KalmanForecaster': {
        #     'output_chunk_length': [ config['forecast_horizon'] ]  #test_length
        #     },
        # 'RNNModel': {
        #     'training_length': [ train_length ]
        #     },
    }
    for mod_name, missingvals in modnames_missingvals.items():
        if mod_name not in config['modnames_grids']:
            continue
        for vname, v in missingvals.items():
            config['modnames_grids'][f'{mod_name}'][f'{vname}'] = v

    return config


def load_models(dir_models):
    """
    Purpose:
        Load pkl models from dir
    Inputs:
        dir_models:
            type: str
            meaning: path to dir where pkl models are loaded from
    Outputs:
        modnames_models:
            type: dict
            meaning: model objs for each modname
    """
    modnames_objtypes = {
        'RNNModel': TorchForecastingModel,
        'TransformerModel': TorchForecastingModel,
        'NBEATSModel': TorchForecastingModel,
        'TCNModel': TorchForecastingModel,
        'VARIMA': ForecastingModel,
        'LightGBMModel': ForecastingModel,
    }

    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    uniques = list(set([pf.split('.')[0] for pf in pkl_files]))

    print(f"Loading {len(uniques)} models...")
    modnames_models = {}
    for uni in uniques:  #for f in pkl_files:
        print(f"  {uni}")
        pkl_path = os.path.join(dir_models, f"{uni}.pkl")
        model = modnames_objtypes[uni].load(pkl_path)
        modnames_models[uni.replace('.pkl', '')] = model
        print(f"    model = {model}")

    return modnames_models


def save_models(modnames_models, dir_out):
    """
    Purpose:
        Save pkl models to dir
    Inputs:
        dir_out:
            type: str
            meaning: path to dir where pkl models are saved to
        modnames_models:
            type: dict
            meaning: model objs for each modname
    Outputs:
        n/a (models saved)
    """
    print(f'Saving {len(modnames_models)} models...')
    for modname, model in modnames_models.items():
        path_out = os.path.join(dir_out, f"{modname}.pkl")
        model.save(path_out)  ## save_data_as_pickle(model, path_out)


def load_pickle_object_as_data(file_path):
    """
    Purpose:
        Load data from pkl file
    Inputs:
        file_path:
            type: str
            meaning: path to pkl file
    Outputs:
        data:
            type: pkl
            meaning: pkl data loaded
    """
    with open(file_path, 'rb') as f_handle:
        data = pickle.load(f_handle)
    return data


def save_data(data_dict, output_dir):
    """
    Purpose:
        Save all data from 'data_dict' to 'dir_out'
    Inputs:
        data_dict:
            type: dict
            meaning: keys are data names ('x_t0', 'y_t0', 'x_t1', 'y_t1', 't0', 't1'), values are pd.DataFrames
        output_dir:
            type: str
            meaning: dir to save csvs too
    Outputs:
        n/a (csv data saved)
    """
    print(f'Saving prepped data...\n  to --> {output_dir}')
    for dname, data in data_dict.items():
        path_out = os.path.join(output_dir, f"{dname}.csv")
        data.to_csv(path_out)


def validate_args(config: dict, data_path, output_dir, data, output_dirs):
    """
    Purpose:
        Ensure validity of args passed to 'run_pipeline'
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
        data:
            type: pd.DataFrame
            meaning: data to run through pipeline
    """
    print("Validating args...")
    # Check which args are found
    found_data = (not isinstance(data, bool))
    found_datapath = (not isinstance(data_path, bool))
    found_outputdir = (not isinstance(output_dir, bool))
    found_outputdirs = (not isinstance(output_dirs, bool))
    print(f"  data = {data}")
    print(f"    found = {found_data}")
    print(f"  data_path = {data_path}")
    print(f"    found = {found_datapath}")
    print(f"  output_dir = {output_dir}")
    print(f"    found = {found_outputdir}")
    print(f"  output_dirs = {output_dirs}")
    print(f"    found = {found_outputdirs}")
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
