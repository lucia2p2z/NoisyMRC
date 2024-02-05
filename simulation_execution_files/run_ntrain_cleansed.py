from base_simulations_ntrain import simulation_ntrain_lrcleansed, simulation_ntrain_mrccleansed
from general_utilities import *
import numpy as np
import os
import sys

if __name__ == '__main__':

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
    rho2 = r2/100  # should be positive and < 0.5
    Nrep = 100
    nvector = np.array(range(1000, 8001, 1000))

    # Parameters MRC related:
    lambda0 = 1
    str_lambda0 = 'l1_'
    det = True  # True = deterministic rule, False =  probabilistic rule

    # Load data:
    X, Y = load_mortality(categorical=categorical)
    if balanced:
        X, Y = balance_data(X, Y, seed=1)
        balanced_type = 'balanced'
    else:
        balanced_type = 'unbalanced'
    print('Balanced dataset: ', balanced)


    # Save the results in appropriate folder
    main_dir = f'../simulation_results/SIM-{Nrep}rep-{balanced_type}/simulation_ntrain/results_0{r1}_0{r2}'
    os.makedirs(main_dir, exist_ok=True)

    # Do simulation for different training size on LOGISTIC REGRESSION
    summary_data1, summary_results1, results_models1, cleanlabinfo1 = simulation_ntrain_lrcleansed(X=X, Y=Y,
                                                                                                   nvector=nvector,
                                                                                                   nrep=Nrep,
                                                                                                   rho1=rho1,
                                                                                                   rho2=rho2)

    print('simulation CLEANSED LABELS on LR done!')

    save_dataframe(
        df=summary_data1,
        df_name='LRCLEANSED_' + dataName + '_summary_data',
        save_dir=main_dir
    )
    save_dataframe(
        df=summary_results1,
        df_name='LRCLEANSED_' + dataName + '_summary_results',
        save_dir=main_dir
    )
    save_dataframe(
        df=results_models1,
        df_name='LRCLEANSED_' + dataName + '_models',
        save_dir=main_dir
    )
    save_dataframe(
        df=cleanlabinfo1,
        df_name='LRCLEANSED_' + dataName + '_cleanlabinfo',
        save_dir=main_dir
    )

    # Do simulation for different training size on MRC
    summary_data2, summary_results2, results_models2, cleanlabinfo2 = simulation_ntrain_mrccleansed(X=X, Y=Y,
                                                                                                    nvector=nvector,
                                                                                                    nrep=Nrep,
                                                                                                    det=det,
                                                                                                    lambda0=lambda0,
                                                                                                    rho1=rho1,
                                                                                                    rho2=rho2)

    print('simulation CLEANSED LABELS on MRC done!')

    save_dataframe(
        df=summary_data2,
        df_name='MRCCLEANSED_' + str_lambda0 + dataName + '_summary_data',
        save_dir=main_dir
    )

    save_dataframe(
        df=summary_results2,
        df_name='MRCCLEANSED_' + str_lambda0 + dataName + '_summary_results',
        save_dir=main_dir
    )

    save_dataframe(
        df=results_models2,
        df_name='MRCCLEANSED_' + str_lambda0 + dataName + '_models',
        save_dir=main_dir
    )
    save_dataframe(
        df=cleanlabinfo2,
        df_name='MRCCLEANSED_' + str_lambda0 + dataName + '_cleanlabinfo',
        save_dir=main_dir
    )





