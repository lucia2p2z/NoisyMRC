import os
from general_utilities import load_dataframe
import pandas as pd
import numpy as np

from plots_utilities import plot_boxplot_cleansed

# Parameters simulations related
fixed_rho1 = False  # True or False
stamp_rho = '010'   # fixed value of rho2 (rho1) if fixed_rho1 = false (true)
Nrep = 100

# Parameters dataset related:
categorical = True
balanced = True

if categorical:
    categorical_type = "alsocat"
else:
    categorical_type = "nocat"
if balanced:
    balanced_type = 'balanced'
else:
    balanced_type = 'unbalanced'
stamp_dataset = f'mortality_{categorical_type}'

# Parameters MRC related
rule = 'det'    # 'det' or 'proba' (determinisitc or probabilistic rule)
l0 = 'l1_'      # value of lambda0

format = 'pdf'  # or 'eps' pr 'png'

if fixed_rho1:
    # Fixed rho1 and varying rho2
    str_varyingrho = 'rho2'
    figure_name = f'rho1_{stamp_rho}.{format}'
else:
    # Fixed rho2 and varying rho1
    str_varyingrho = 'rho1'
    figure_name = f'rho2_{stamp_rho}.{format}'

# Set the main directory to save the plots
plot_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_perfeval/my_plots_{categorical_type}_{rule}/boxplot_cleansed'

ntrain = 8000
plot_dir = plot_dir + f'_ntrain8k/'
os.makedirs(plot_dir, exist_ok=True)

df = pd.DataFrame(
    {'value': {},
     'type': {},
     'classifier': {},
     str_varyingrho: {}
     })

for varyingrho in ['010', '025', '040']:

    r = float('0.' + varyingrho) * 10

    if fixed_rho1:
        perfeval_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_perfeval/results_{stamp_rho}_{varyingrho}'
        ntrain_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_ntrain/results_{stamp_rho}_{varyingrho}'
    else:
        perfeval_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_perfeval/results_{varyingrho}_{stamp_rho}'
        ntrain_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_ntrain/results_{varyingrho}_{stamp_rho}'

    # Load the important results

    # ___________________ MRC _____________________________________

    data_mrc = load_dataframe(
        load_dir=ntrain_dir,
        df_name='MRC_' + l0 + stamp_dataset + '_summary_results',
        saved_format='gzip'
    )
    summary_data_mrc = load_dataframe(                # I need this just for ntrain
        load_dir=perfeval_dir,
        df_name='MRC_' + l0 + stamp_dataset + '_summary_data',
        saved_format='gzip'
    )

    data_mrc = data_mrc.drop(labels=['times'], axis=0)
    data_mrc = data_mrc.drop(labels=['mrc_clean', 'mrc_nocorr'], axis=1)

    # ___________________ NATARAJAN _____________________________________

    data_lr_nata = load_dataframe(
        load_dir=perfeval_dir,
        df_name='NATA_' + stamp_dataset + '_summary_results',
        saved_format='gzip'
    )
    data_nata = load_dataframe(
        load_dir=ntrain_dir,
        df_name='NATA_' + stamp_dataset + '_summary_results',
        saved_format='gzip'
    )

    data_lr_nata = data_lr_nata.drop(labels='lr_nocorr', axis=1)
    data_nata = pd.concat([data_nata, data_lr_nata], axis=0, ignore_index=False)
    data_nata = data_nata.drop(labels=['times', 'biased_loss'], axis=0)

    # ___________________ LR CLEANSED LAB. _____________________________________

    data_cleansed = load_dataframe(
        load_dir=perfeval_dir,
        df_name='LRCLEANSED_' + stamp_dataset + '_summary_results',
        saved_format='gzip'
    )

    data_cleansed = data_cleansed.drop(labels=['unbiased_loss', 'times'], axis=0)

    # ________________________________________________________________________

    nvector = summary_data_mrc.ntrain_vector[0]
    mrc_err = data_mrc['mrc_back']['errors']
    nata_err = data_nata['natarajan']['errors']
    lr_err = data_cleansed['lr_cleansed']['errors']

    mrc_bound = data_mrc['mrc_back']['bounds']
    nata_ule = data_nata['natarajan']['unbiased_loss']
    lr_ble = data_cleansed['lr_cleansed']['biased_loss']

    index_ntrain = np.where(nvector == ntrain)[0]
    loss = 'errors'

    df = df._append(
        {'value': mrc_err[:, index_ntrain],
         'type': loss,
         'classifier': 'mrc_back',
         str_varyingrho: r,
         }, ignore_index=True)
    df = df._append(
        {'value': nata_err[:, index_ntrain],
         'type': loss,
         'classifier': 'natarajan',
         str_varyingrho: r,
         }, ignore_index=True)
    df = df._append(
        {'value': lr_err[:, index_ntrain],
         'type': loss,
         'classifier': 'lr_cleansed',
         str_varyingrho: r,
         }, ignore_index=True)

    df = df._append(
        {'value': mrc_bound[:, index_ntrain],
         'type': 'minimax',
         'classifier': 'mrc_back',
         str_varyingrho: r,
         }, ignore_index=True)
    df = df._append(
        {'value': nata_ule[:, index_ntrain],
         'type': 'ULE',
         'classifier': 'natarajan',
         str_varyingrho: r,
         }, ignore_index=True)
    df = df._append(
        {'value': lr_ble[:, index_ntrain],
         'type': 'BLE',
         'classifier': 'lr_cleansed',
         str_varyingrho: r,
         }, ignore_index=True)


df = df.explode('value')
df.value = df.value.astype('float')

df_noerr = df.loc[df['type'] != 'errors']
df_err = df.loc[df['type'] == 'errors']

err_column = np.asarray(df_err['value'])
df_tot = df_noerr
df_tot.loc[:, 'errors'] = err_column

plot_boxplot_cleansed(data=df_tot, str_varyingrho=str_varyingrho,
                      plot_dir=plot_dir, title='', figure_name=figure_name, format=format)

