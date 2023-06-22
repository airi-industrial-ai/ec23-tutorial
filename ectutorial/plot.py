import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_risk(mn_log, lb_log, ub_log, err_test, title):
    plt.figure(figsize=(7, 3), layout='constrained')
    plt.plot(1 - np.arange(1, 10) / 10, [np.sqrt(i) for i in mn_log], label='Risk')
    plt.fill_between(
        1-np.arange(1, 10) / 10, 
        [np.sqrt(i) for i in lb_log], 
        [np.sqrt(i) for i in ub_log], 
        color='b', 
    alpha=.1
    )
    plt.hlines([np.sqrt(err_test.mean())], xmin=0.1, xmax=0.9, color="C1", label='Test')

    plt.xlabel('Subpopulation Size (1-α)')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.show()
