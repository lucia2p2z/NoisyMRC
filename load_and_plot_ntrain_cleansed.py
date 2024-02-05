from general_utilities import load_dataframe
from plots_utilities import plot_ntrain_cleansed
import os


# Parameters simulations related
stamprhos = '010_040'
Nrep = 100

format = 'pdf'  # or 'eps' pr 'png'

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

figure_name = f'{stamprhos}.{format}'
# Set the directory to find the data
main_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_ntrain/results_{stamprhos}'
# Set the main directory to save the plots
plot_dir = f'./simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_ntrain/my_plots_{categorical_type}_{rule}'
os.makedirs(plot_dir, exist_ok=True)

# Load summary dataset
summary_data_mrc = load_dataframe(
    load_dir=main_dir,
    df_name='MRC_' + l0 + stamp_dataset + '_summary_data',
    saved_format='gzip'
)

# Load the important results
res_mrc = load_dataframe(
    load_dir=main_dir,
    df_name='MRC_' + l0 + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_mrccleansed = load_dataframe(
    load_dir=main_dir,
    df_name='MRCCLEANSED_' + l0 + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_lrcleansed = load_dataframe(
    load_dir=main_dir,
    df_name='LRCLEANSED_' + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_nata = load_dataframe(
    load_dir=main_dir,
    df_name='NATA_' + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_mrc_est = load_dataframe(
    load_dir=main_dir,
    df_name='MRCest_' + l0 + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

nvector = summary_data_mrc.ntrain_vector[0]

plot_ntrain_cleansed(nvector, res_mrc, res_mrccleansed, res_nata, res_lrcleansed,
                     plot_dir=plot_dir, title='', figure_name=figure_name, shade=True, format=format)
