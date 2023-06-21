import numpy as np
import seaborn as sns
import pandas as pd

def plot_pair_fake_val(fake, val_data, scaler):
    seq_len = len(fake)
    _fake_plot = pd.DataFrame(
        scaler.inverse_transform(fake),
        index=val_data.index[:seq_len],
        columns=val_data.columns,
    )
    _fake_plot['source'] = 'fake'

    np.random.seed(0)
    _val_plot = val_data.iloc[np.random.randint(len(val_data), size=seq_len)].copy()
    _val_plot['source'] = 'val'

    sns.pairplot(
        pd.concat([_fake_plot, _val_plot]),
        hue='source',
        corner=True,
        plot_kws={'edgecolor': 'none', 'alpha': 0.3, 's': 10},
    )
