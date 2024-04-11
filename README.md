# Noisy Minimax Risk Classifier 

[![Made with!](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](#python-code)

This repository is the implementation of the method presented in "**Minimax Risk Classifiers for Mislabeled Data:
a study on Patient Outcome Prediction Tasks**". 

The algorithm proposed in the paper provides an efficient method to learn from noisy labels and a robust method to evaluate the performance of the classifier, even in scenarios where clean test data are not available. 
This algorithm can be used whether the transition matrix $T$ - representing the noise - is known or not (in this last case the proposed algorithm exploits an external library to estimate it from the data). 

## Requirements
- `Python` >= 3.6 (the code was developed using `Python` = 3.11)
- `numpy`, `scipy`, `scikit-learn`, `cvxpy`, `mosek`, `gurobipy`, `pandas`, `cleanlab`


### Additional Requirements
- Depending on the version of `Python` installed in your environment, you may need to install [`CMake`](https://cmake.org). If prompted, you can install it following the guidelines in [Download CMake](https://cmake.org/download/). 
- The implementation of the proposed algorithm based on CVXpy uses MOSEK optimizer, which requires a license. You can get a free academic license from [here](https://www.mosek.com/products/academic-licenses/).


## How to install

To install the required libraries do as follow:
 1) Install the standard libraries listed in the `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```
2) Run the following commands to install the paper's custom distribution of the MRCpy library:
    ```
    cd MRCpy
    python3 setup.py install
    ```

## Data 
The `data_mortality` folder contains the *ICU Mortality* dataset, polished as explained in the associated paper. In particular, it provides two version of the dataset in CSV format:
1. `mortality_alsocat.csv`: Contains the polished data with all the features.
2. `mortality_nocat.csv`: Contains the polished data without categorical variables.

The original dataset is available [here](https://www.kaggle.com/c/widsdatathon2020/data) (a login is needed to download it).

The `datasets` folder contains *Mammographic Mass* datasets, as well as the additional ones mentioned in the Appendices of the paper.


**NOTE**: Please ensure that the folders remains in their current location within the parent directory. If you choose to relocate the folder, remember to update the file paths accordingly.


## Experiments

The files in the ``simulation_execution_files`` folder contain the scripts to replicate the experiments of the paper. 
Experiments to _learn on noisy_ data and _evaluate on clean_ test data are: 
- $T$ known experiments: 
    - `run_ntrain_mrc.py`: performs training and evalution of **NoisyMRC**, **NaiveMRC**, and **OracleMRC**;
    - `run_ntrain_nata.py`: performs training and evalution of **Noisy LR**;
    - `run_ntrain_lr.py`: performs training and evalution of **Naive LR**, and **Oracle LR**;
    - `run_ntrain_cl.py`: performs training and evalution of the method **CleanLearning**.
- $T$ unknown experiments: 
    - `run_ntrain_mrcest.py`:performs training and evalution of **NoisyMRC** on $T$ estimated;
    - `run_ntrain_cleansed.py`: performs training and evalution of **Cleansed MRC**, and **Cleansed LR**.

Experiments to _learn and evaluate on noisy_ data:     
- `run_perfeval_mrc.py`: performs training of **Noisy MRC** and evalution of it with **Minimax**;
- `run_perfeval_nata.py`: performs training of **Noisy LR** and evalution of it with **ULE**;
- `run_perfeval_lrcleansed.py`: performs training of **Cleansed LR** and evalution of it with **LE**.

### Parameters:

Inside the Python scripts listed above you can manually set various parameters for training. Among which:
#### Parameters MRC related: (when needed)
- `lambda0`: *(float, defalut: 1)* Defines the parameter $\lambda_0$ of the MRCs.
- `det`: *(boolean, default: True)*  If set to True uses the determininstc rule in the MRCs. 

#### Parameters dataset related:
- `categorical`: *(boolean, default: True)* If set ot True, reads the dataset containing also the categorical variables. Otherwise, read the dataset where they have been removed, keeping only the continuous features. 
- `balanced`: *(boolean, default: True)* If set to True, balaces the dataset (to have the same percentage of 0's and 1's labels).

###### Parameter simulation related:
- `r1`: *(int, contraints: > 0, <50)* Specifies the percentage of 0's labels mislabeled. In particular it specifies the value of the noise rate $\rho_1$ ($\rho_1=$`r1`$/100$);
- `r2`: *(int, contraints: > 0, <50)* Specifies the percentage of 1's labels mislabeled. In particular it specifies the value of the noise rate $\rho_2$ ($\rho_2=$`r2`$/100$);
- `Nrep`: *(int)* Specifies the number of repetitions of the simulation.
- `nvector`:  *(numpy array)* Specifies the training sizes to use.



## Replicating the plots in the submission

To replicate the plots presented in the submission for *ICU Mortality* dataset, you will need to use the Python files named `load_and_plot_***`. These files are designed to load the results and generate various plots. Below are instructions for reproducing specific figures:

-  To reproduce **Figure 1**: run `load_and_plot_ntrain.py`.
-  To reproduce **Figure 2**: run `load_and_plot_boxplot.py`.
-  To reproduce **Figure 3**: run `load_and_plot_ntrain_est.py`.
-  To reproduce **Figure 4**: run `load_and_plot_ntrain_cleansed.py`.

Ensure that you have the necessary data before running these scripts.
