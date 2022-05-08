from cgi import test
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as etree
import logging as log


xml_path = './data/OhioT1DM'
processed_path = './data/processed'


years = ['2018', '2020']
modes = ['train', 'test']

os.makedirs(processed_path, exist_ok=True)

patient_ids = []

feas = ['finger_stick', 'basal', 'temp_basal',
        'bolus', 'meal', 'glucose_level']

base_time = pd.Timestamp('2018-01-01')


def make_sequence_data(df):
    def _merge_rec(x):
        return x.mean(axis=0)

    def _set_index(x, t_cols=['ts'], val_cols=None):
        for t_col in t_cols:
            x[t_col] = pd.to_datetime(x[t_col], dayfirst=True)
        if 'ts' in x.columns:
            x = x.set_index('ts').sort_index()
        if val_cols is not None:
            x = x.rename(columns={val_cols[0]: val_cols[1]})
        return x

    def _set_temp_basal(basal, temp_basal):
        for row in temp_basal.itertuples():
            ts_begin = row.ts_begin
            ts_end = row.ts_end
            basal.loc[ts_begin: ts_end] = row.value
        return basal

    finger_stick = _set_index(
        df['finger_stick'], val_cols=['value', 'finger_stick']
    )

    cgm = _set_index(
        df['glucose_level'], val_cols=['value', 'cgm']
    )

    basal = _set_index(df['basal'], val_cols=['value', 'basal'])

    if df['temp_basal'].shape[0] > 0:
        temp_basal = _set_index(
            df['temp_basal'], t_cols=['ts_begin', 'ts_end']
        )
        basal = _set_temp_basal(basal, temp_basal)

    bolus = df['bolus'].rename(columns={'ts_begin': 'ts'})[['ts', 'dose']]
    bolus = _set_index(bolus, t_cols=['ts'], val_cols=['dose', 'bolus'])

    if df['meal'].shape[0] > 0:
        meal = _set_index(df['meal'], val_cols=['carbs', 'meal'])[['meal']]
    else:
        # remove
        return None

    data = pd.concat([finger_stick, basal, bolus, meal, cgm]).sort_index()
    data = data.astype('float')

    data = data.reset_index()
    data['ts'] = data['ts'].map(
        lambda x: (x - base_time).total_seconds() // 60
    )

    data = data.set_index('ts').sort_index()

    data['cgm'] = data['cgm'].fillna(method='ffill')

    data.loc[~pd.isnull(data['finger_stick']), 'finger_stick'] = \
        data.loc[~pd.isnull(data['finger_stick']), 'cgm']

    data = data.groupby(level=0).apply(lambda x: _merge_rec(x))

    return data


def main():
    for year in years:
        for mode in modes:
            xml_full_path = os.path.join(xml_path, year, mode)
            print('Processing {}...'.format(xml_full_path))
            for xml_file in os.listdir(xml_full_path):
                _xml_file = os.path.join(xml_full_path, xml_file)
                print(_xml_file)
                tree = etree.parse(_xml_file)
                root = tree.getroot()

                patient_id = root.attrib.get('id')
                patient_ids.append(patient_id)

                df = {}

                for fea in feas:
                    df[fea] = pd.DataFrame(
                        [x.attrib for n in tree.iter(fea) for x in n]
                    )

                df = make_sequence_data(df)

                if df is not None:
                    df.to_csv(f'./data/processed/{patient_id}_{mode}.csv')

    for p_id in [540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596]:
        train_file = './data/processed/{}_train.csv'.format(p_id)
        if not os.path.exists(train_file):
            train_df = None
        else:
            train_df =  pd.read_csv(train_file)
            os.remove(train_file)

        test_file = './data/processed/{}_test.csv'.format(p_id)
        if not os.path.exists(test_file):
            test_df = None
        else:
            test_df = pd.read_csv(test_file)
            os.remove(test_file)

        df = pd.concat([train_df, test_df])
        if df is not None:
            df.set_index('ts').to_csv(f'./data/processed/{p_id}.csv')
if __name__ == '__main__':
    main()
