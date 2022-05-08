import os

import pandas as pd
import numpy as np
import ipdb

state_mean = np.array([5.774167109012415, 4.980055120209744, 3.3048831379042, 5.4111882346560085, 4.63773546098587, 2.1886929342054104,
                       2.4791117362502604, 5.012958395624457, 4.978418538049152, 1.6112976531080372, 4.827765178738924, 4.687232183059351, 5.420535468575779])


state_std = np.array([3.9335407475656656, 3.1196996593347244, 2.7874556445207386, 0.3369634577178862, 0.3300954394840192, 0.3149589446582321,
                      2.0470918897863215, 0.3090696903591714, 0.2807260148170816, 0.2940418665709757, 0.772049271297565, 0.3498235085462231, 0.3307852617213216])



def _get_inputs_outputs(lookback, lookahead, fea_cols, filepath):
    inputs = []
    outputs = []

    start_time = pd.Timestamp('2018-01-01')

    def _extract(data):
        data['Time'] = data['Time'].map(
            lambda x: (x - start_time).total_seconds() // 60 
        )
        data = data.rename(
            columns={
                'Time': 'ts',
                'BG': 'finger_stick',
                'CHO': 'meal',
            }
        )

        # U/min -> U/hr
        data['basal'] = data['basal'] * 60
        data['finger_stick'] = data['finger_stick'] / 100.0

        data['meal'] = data['meal'].replace(0.0, np.nan)
        data['filled_meal'] = data['meal'].fillna(
            method='ffill').fillna(0.0)

        data['bolus'] = data['bolus'].replace(0.0, np.nan)

        data = data.set_index('ts').sort_index()

        data['basal'] = data['basal'].fillna(method='ffill')
        data['filled_meal'] = data['meal'].fillna(method='ffill')

        data['filename'] = -1

        state_cols = ['state_{}'.format(x) for x in range(13)]

        data = data[fea_cols + ['filename'] + state_cols]

        data[state_cols] = (data[state_cols].values - state_mean) / state_std

        # ipdb.set_trace()

        for i in range(lookback, data.shape[0]):
            _input = data.iloc[i - lookback: i]
            _output = data.iloc[i: i + lookahead]

            if _input.shape[0] != lookback or _output.shape[0] != lookahead:
                continue
            
            _input = _input.reset_index()
            _output = _output.reset_index()

            if pd.isnull(_output['finger_stick']).sum() < _output.shape[0]:
                if _output.shape[0] > 0:
                    inputs.append(_input.values)
                    outputs.append(_output.values)

    files = os.listdir(filepath)
    for _file in files:
        data = pd.read_csv(
            os.path.join(filepath, _file),
            parse_dates=['Time']
        )
        _extract(data)

    # convert the input to an array
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    return inputs, outputs

if __name__ == '__main__':
    inputs, outputs = _get_inputs_outputs(12, 6, [
    'finger_stick', 'meal', 'filled_meal', 'basal', 'bolus'
], './simulation/data')
    print(inputs.shape, outputs.shape)
