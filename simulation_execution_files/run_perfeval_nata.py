from base_simulations_perfeval import simulation_perfeval_lr_nata
from general_utilities import *
import numpy as np
import os
import sys


# Parameters dataset related:
categorical = True
balanced = False
if categorical:
    dataName = "mortality_alsocat"
else:
    dataName = "mortality_nocat"

# Parameter simulation related
r1 = 10
r2 = 10
rho1 = r1/100   # should be positive and < 0.5
rho2 = r2/100   # should be positive and < 0.5
Nrep = 100
nvector = np.array(range(1000, 8001, 1000))

# Load data:
X, Y = load_mortality(categorical=categorical)
if balanced:
    X, Y = balance_data(X, Y, seed=1)
    balanced_type = 'balanced'
else:
    balanced_type = 'unbalanced'
print('Balanced dataset: ', balanced)

# Do simulation for different training size
summary_data, summary_results, results_models = simulation_perfeval_lr_nata(X=X, Y=Y,
                                                                            nvector=nvector,
                                                                            nrep=Nrep,
                                                                            rho1=rho1,
                                                                            rho2=rho2)
print('simulation perfeval NATARAJAN done!')

# Save the results in appropriate folder
main_dir = f'../simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_perfeval/results_0{r1}_0{r2}'
os.makedirs(main_dir, exist_ok=True)

filename1 = 'NATA_' + dataName + '_summary_data'
save_dataframe(
    df=summary_data,
    df_name=filename1,
    save_dir=main_dir
)

filename2 = 'NATA_' + dataName + '_summary_results'
save_dataframe(
    df=summary_results,
    df_name=filename2,
    save_dir=main_dir
)


