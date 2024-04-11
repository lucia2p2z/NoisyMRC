""" Set of loaders and convenient functions to access Dataset.
=========================================================
"""
import csv
import zipfile
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch


def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y


def load_adult(with_info=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'adult.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)
        # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_diabetes(with_info=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_ecoli(with_info=False):
    """Load and return the Ecoli Dataset (classification).

    =================== =========================
    Classes                                     8
    Samples per class     [143,77,52,35,20,5,2,2]
    Samples total                             336
    Dimensionality                              8
    Features                 int, float, positive
    =================== =========================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'ecoli.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'ecoli.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray([float(i) for i in d[1:-1]], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)



def load_credit(with_info=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'credit.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)


def load_magic(with_info=False):
    """Load and return the Magic Gamma Telescope dataset (classification).

    =================== ======================
    Classes                                 2
    Samples per class            [6688,12332]
    Samples total                       19020
    Dimensionality                         10
    Features                            float
    =================== ======================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'magic.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'magic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data,
                 target=normalizeLabels(target),
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)



def load_haberman(with_info=False):
    """Load and return the Haberman's Survival Data Set (classification).

    ================== ============
    Classes                      2
    Samples per class    [225, 82]
    Samples total              306
    Dimensionality               3
    Features                   int
    ================== ============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of haberman csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'haberman.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'haberman.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=descr_text,
                 feature_names=['PatientAge',
                                'OperationYear',
                                'PositiveAxillaryNodesDetected'])


def load_mammographic(with_info=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ================== ============
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ================== ============

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    # fdescr_name = join(module_path, 'descr', 'mammographic.rst')
    # with open(fdescr_name) as f:
    #     descr_text = f.read()

    data_file_name = join(module_path, 'data', 'mammographic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i, d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 DESCR=None,
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'])


def load_indian_liver(with_info=False):
    """Load and return the Indian Liver Patient Data Set
    (classification).

    ============================ =============================
    Classes                                                 2
    Samples per class                              [416, 167]
    Samples total                                         583
    Dimensionality                                         10
    Features                                       int, float
    Missing Values                                     4 (nan)
    =========================== ==============================

    Parameters
    ----------
    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``with_info`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'indianLiverPatient.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)
    # with open(join(module_path, 'descr',
    #                'indianLiverPatient.rst')) as rst_file:
    #     fdescr = [line.decode('utf-8').strip() \
    #               for line in rst_file.readlines()]

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if not with_info:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=['Age of the patient',
                                'Gender of the patient',
                                'Total Bilirubin',
                                'Direct Bilirubin',
                                'Alkaline Phosphotase',
                                'Alamine Aminotransferase',
                                'Aspartate Aminotransferase',
                                'Total Protiens',
                                'Albumin',
                                'A/G Ratio'])

