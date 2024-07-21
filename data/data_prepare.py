import pandas as pd
import numpy as np
import random
import logging
from sklearn.model_selection import train_test_split


class Dataset:
    RAW = 'raw'
    VELOCITY = 'velocity'
    RAW_VELOCITY = 'raw_velocity'


class Target:
    SLUDGE = 'sludge'
    DIRECTION = 'direction'


log = logging.getLogger('DataPrepare')


def get_valid_wells(data, well_name_column, depth_column):
    wells = []
    for well_name in data[well_name_column].unique():
        well_data = data[data[well_name_column] == well_name].sort_values(by='depth')
        well_data[depth_column] = well_data[depth_column].diff()
        unique_depth = well_data[depth_column].unique()
        valid = True
        for depth in unique_depth:
            if depth > 30:
                valid = False
                break
        well = {'valid': valid, 'name': well_name, 'depths': unique_depth, 'wellid': well_data.iloc[0]['wellid']}
        wells.append(well)
    wells_df = pd.DataFrame(wells)
    log.debug(str(wells_df[~wells_df['valid']]))
    return wells_df[wells_df['valid']]['name'].tolist()


def fill_nans(data):
    rows = data.shape[0]
    nans = data.isna().sum() / rows * 100
    for column in nans.index:
        if nans[column] == 0:
            continue
        if nans[column] < 50:
            median = data[column].median()
            data.fillna({column: median}, inplace=True)
            continue
        data.drop(column, axis=1, inplace=True)


def make_datasets(data, target_column, well_name_column):
    X = data.drop([target_column, well_name_column], axis=1)
    y = data[target_column]
    random_dataset = train_test_split(X, y, random_state=30)
    all_well_names = data[well_name_column].unique()
    random.shuffle(all_well_names)
    test_well_names = all_well_names[-10:]
    log.debug(f'Test Well Names {test_well_names}')
    well_train_data = data[~data[well_name_column].isin(test_well_names)]
    well_test_data = data[data[well_name_column].isin(test_well_names)]
    wells_dataset = well_train_data.drop([target_column, well_name_column], axis=1), well_test_data.drop(
        [target_column, well_name_column], axis=1), well_train_data[target_column], well_test_data[target_column]
    datasets = {
        # 'random shuffle dataset': random_dataset,
        'wells shuffle dataset': wells_dataset
    }
    return datasets


def calculate_velocities(data, data_columns, well_name_column, depth_column):
    data = data.sort_values(by=depth_column)
    for col in data_columns:
        data[f'{col}_diff'] = data.groupby(well_name_column)[col].diff()
    return data.fillna(0)


def calculate_new_target(data, well_name_column, target_column, target_name):
    sludge_names = data[target_name].unique()
    sludge_order = {
        'Fr': 1,
        'Fr низ': 2,
        'Fr подошв': 3,
        'Bzh_6': 4,
        'Bzh_5b': 5,
        'Bzh_5a': 6,
        'Bzh_4b': 7,
        'Bzh_4a': 8,
        'Bzh_3': 9,
        'Bzh_2b верх': 10,
        'Bzh_2b низ': 11,
        'Bzh_2a верх': 12,
        'Bzh_2a низ': 13,
        'Bzh_1': 14,
        'Ab': 15
    }
    sludge_ids_order = {}
    inverted_sludge_order = {}
    sludge_ids = data[[target_column, target_name]]
    for sludge_name in sludge_names:
        sludge = sludge_ids[sludge_ids[target_name] == sludge_name].iloc[0]
        sludge_ids_order[sludge[target_column]] = sludge_order[sludge[target_name]]
        inverted_sludge_order[sludge_order[sludge[target_name]]] = sludge[target_column]
    all_well_names = data[well_name_column].unique()
    data['target_order'] = data[target_column].apply(lambda id: sludge_ids_order[id])
    data['new_target'] = np.nan
    exclude_wells = []
    for well_name in all_well_names:
        well_data = data[data[well_name_column] == well_name].sort_values('depth')
        well_data['new_target'] = well_data['target_order'].diff()
        found_strange_unique = False
        for unique in well_data['new_target'].unique():
            if unique == unique and abs(unique) > 1:
                exclude_wells.append(well_name)
                found_strange_unique = True
                break
        if found_strange_unique:
            continue
        data[data[well_name_column] == well_name] = well_data
    log.debug(f'Excluded wells {exclude_wells}')
    data = data[~data[well_name_column].isin(exclude_wells)]
    data.fillna({'new_target': 0}, inplace=True)
    data['result'] = np.nan
    for well_name in data[well_name_column].unique():
        well_data = data[data[well_name_column] == well_name].sort_values('depth')
        previous_index = None
        for number, current_index in enumerate(well_data.index):
            if number == 0:
                well_data.loc[current_index, 'result'] = sludge_ids_order[well_data.loc[current_index, target_column]]
            else:
                well_data.loc[current_index, 'result'] = well_data.loc[previous_index, 'result'] + well_data.loc[
                    current_index, 'new_target']
            previous_index = current_index
        data[data[well_name_column] == well_name] = well_data
    log.debug(f'This must be True: {data[target_column].equals(data["result"].apply(lambda result: inverted_sludge_order[result]))}')
    data = data.drop(['result', 'target_order', target_column, target_name], axis=1)
    target_column = 'new_target'
    return data, target_column


def prepare_dataset(dataset_path, dataset_type: Dataset = Dataset.RAW, target_type: Target = Target.SLUDGE):
    data = pd.read_csv(dataset_path)
    well_name_column = 'well_name'
    depth_column = 'depth'
    target_column = 'stratigraphy_expert_id'
    target_name = 'stratigraphy_expert_name'
    drop_columns = ['probe_num', 'wellid']
    property_columns = [well_name_column, depth_column, target_column, target_name, 'probe_num', 'wellid']
    data = data[data[well_name_column].isin(get_valid_wells(data, well_name_column, depth_column))]
    data = data.drop(drop_columns, axis=1)
    fill_nans(data)
    data_columns = list(
        filter(lambda column: column not in property_columns, data.columns)
    )
    if dataset_type == Dataset.RAW_VELOCITY:
        data = calculate_velocities(data, data_columns, well_name_column, depth_column)
    if dataset_type == Dataset.VELOCITY:
        data = calculate_velocities(data, data_columns, well_name_column, depth_column)
        data = data.drop(data_columns, axis=1)
    if target_type == Target.DIRECTION:
        data, target_column = calculate_new_target(data, well_name_column, target_column, target_name)
    else:
        data = data.drop(target_name, axis=1)
    return make_datasets(data, target_column, well_name_column)
