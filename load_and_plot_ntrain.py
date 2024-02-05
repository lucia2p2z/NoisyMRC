from general_utilities import load_dataframe
from plots_utilities import plot_ntrain_all_methods
from plots_utilities import plot_ntrain_corrected
import os

# Parameters simulations related
stamprhos = '010_010'
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

res_lr = load_dataframe(
    load_dir=main_dir,
    df_name='LR_' + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_nata = load_dataframe(
    load_dir=main_dir,
    df_name='NATA_' + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

res_cl = load_dataframe(
    load_dir=main_dir,
    df_name='CL_' + stamp_dataset + '_summary_results',
    saved_format='gzip'
)

nvector = summary_data_mrc.ntrain_vector[0]

plot_ntrain_all_methods(nvector, data_mrc=res_mrc, data_lr=res_lr, data_nata=res_nata, data_cl=res_cl,
                         plot_dir=plot_dir, title='', figure_name=figure_name, shade=True, format=format)

plot_ntrain_corrected(nvector, data_mrc=res_mrc, data_nata=res_nata, data_cl=res_cl,
                      plot_dir=plot_dir, title='', figure_name=figure_name, shade=True, format=format)


