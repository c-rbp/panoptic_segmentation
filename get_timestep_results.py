import os
import numpy as np
import pandas as pd


def get_results(path, f='log.txt'):
    filepath = os.path.join(path, f)
    next_line = False
    data = []
    ts = []
    pq = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            if 'TIMESTEPS: ' in line and 'TAU' in prev_line:
                data.append(float(line.split('TIMESTEPS: ')[1].strip('\n')))
                ts.append(float(line.split('TIMESTEPS: ')[1].strip('\n')))
            elif 'PQ,SQ,RQ,PQ_th,SQ_th,RQ_th,PQ_st,SQ_st,RQ_st' in line:
                next_line = True
            elif next_line:
                next_line = False
                data.append(float(line.split('copypaste: ')[1].split(',')[0]))
                pq.append(float(line.split('copypaste: ')[1].split(',')[0]))
            prev_line = line
    # df = pd.DataFrame(np.stack((ts[1:31], pq), 1), columns=['timesteps', 'PQ'])
    if len(ts) > 60:
        df = pd.DataFrame(np.stack((ts[1:31], pq), 1), columns=['timesteps', 'PQ'])
    else:
        df = pd.DataFrame(np.stack((ts[:30], pq), 1), columns=['timesteps', 'PQ'])
    df['path'] = path
    return df


if __name__ == '__main__':
    dirs = [
        'RFPN_cbp20-tstest',
        'RFPN_bptt1-tstest',
        'RFPN_bptt3-tstest',
        'RFPN_bptt5-tstest',
    ]
    dfs = []
    for d in dirs:
        dfs.append(get_results(os.path.join('outputs', d)))
    df = pd.concat(dfs)
    np.save('ts_data', df)
    print(df)
