import os
import pandas as pd
import numpy as np
import ipdb


def random_sample(inputs, outputs, p=0.1):
    if isinstance(p, float) and p < 1.0:
        np.random.seed(0)
        idx = np.arange(len(inputs))
        idx = np.random.choice(idx, int(len(inputs) * p), )
        return inputs[idx], outputs[idx]
    elif p > 500:
        p = str(int(p))
        inputs = np.asarray([x for x in inputs if x[-1][-1] == p])
        outputs = np.asarray([x for x in outputs if x[-1][-1] == p])
        return inputs, outputs
    elif p == 1.0:
        return inputs, outputs


def _get_inputs_outputs(lookback, lookahead, fea_cols, filepath, frac=1.0):
    inputs = []
    outputs = []

    def _extract(data, filename):
        data = data.sort_values(by='ts')
        data = data.set_index('ts').sort_index()
        
        data = data[['finger_stick', 'meal',
                     'basal', 'bolus']].dropna(how='all')

        data['finger_stick'] = data['finger_stick'] / 100.0

        data['basal'] = data['basal'].fillna(method='ffill')
        data['filled_meal'] = data['meal'].fillna(method='ffill')

        data['filename'] = filename
        data = data[fea_cols + ['filename']]

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
            os.path.join(filepath, _file)
        )
        data['ts'] = data['ts'].astype('int')
        _extract(data, filename=_file.split('.')[0])

    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs)

    inputs, outputs = random_sample(inputs, outputs, p=frac)

    return inputs, outputs

if __name__ == '__main__':
    inputs, outputs = _get_inputs_outputs(12, 6, [
    'finger_stick', 'meal', 'filled_meal', 'basal', 'bolus'
], './real/data/processed', 0.1)
    print(inputs[0], outputs.shape)
