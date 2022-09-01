import pandas as pd

def split_data(data, data_cap: int, time_col: str, features_inout: dict, test_prop: float):
    data = data[:data_cap]
    train_prop = 1-test_prop
    print(f'Spltting data Time0 / Time1 by {int(train_prop*100)} / {int(test_prop*100)}...')
    t0_endrow = int(data.shape[0]*train_prop)
    features_all = list(set(features_inout['in']+features_inout['pred']))
    if time_col:
        features_all += [time_col]
    # TODO: add time_col
    t0, t1 = data[:t0_endrow][features_all], data[t0_endrow:][features_all]
    x_t0, y_t0 = t0[features_inout['in']+[time_col]], t0[features_inout['pred']+[time_col]]
    x_t1, y_t1 = t1[features_inout['in']+[time_col]], t1[features_inout['pred']+[time_col]]
    dict_data = {'x_t0':x_t0, 'x_t1':x_t1, 'y_t0':y_t0, 'y_t1':y_t1, 't0':t0, 't1':t1, 't0t1': pd.concat([t0,t1], axis=0)}
    for dname, d in dict_data.items():
        print(f"  {dname} = {d.shape}")
    return dict_data
