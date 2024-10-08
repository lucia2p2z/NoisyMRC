.. MRCpy documentation master file, created by
   sphinx-quickstart on Sat Jun  5 15:07:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MRCpy: A Library for Minimax Risk Classifiers
=============================================
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


|Travis-CI Build Status| |Code coverage|

`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ library implements minimax risk classifiers (MRCs) that are based on robust risk minimization and can utilize 0-1-loss. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. The presented library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines. In addition, MRCpy implements recent feature mappings such as Fourier, ReLU, and threshold features.

`MRCpy <https://github.com/MachineLearningBCAM/MRCpy>`_ library incorporates a variety of datasets, along with descriptions and convenient loader functions for each dataset. More information about loaders is available in Dataset Loaders :ref:`load`. The next datasets from the `UCI Repository <http://archive.ics.uci.edu/ml/index.php>`_ are available: credit, diabetes, ecoli, glass, haberman, indian liver patient, iris, letter recognition, mammographic, optdigits, redwine, satellite and segment. 

There also be several datasets realated to computer vision which are actually "feature datasets". We obtained these features by using a pretrained neural network over the images and taking the features from the second last layer. You can check more about this in our examples :ref:`featureextraction` and :ref:`feature_mrc`.
The image dataset `Yearbook dataset <https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html>`_ is available both in its original version consisting of portrait images and in extracted features form in a CSV file. There are also feature datasets of `MNIST <http://yann.lecun.com/exdb/mnist/>`_ and `Cats vs Dogs <https://www.microsoft.com/en-us/download/details.aspx?id=54765>`_ datasets which images versions are directly available using `Tensorflow Datasets <https://www.tensorflow.org/datasets/catalog/overview>`_. For all these feature datasets we used a pretrained ResNet18 over ImageNet.



Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   api

.. toctree::
   :maxdepth: 2

   auto_examples/index


References
----------

 For more information about the MRC method and the MRCpy library, one can refer to the following resources:

   - [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020). Minimax Classification with 0-1 Loss and Performance Guarantees. Advances in Neural Information Processing Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

         ::
         
               @article{mazuelas2020minimax,
                  title={Minimax Classification with 0-1 Loss and Performance Guarantees},
                  author={Mazuelas, Santiago and Zanoni, Andrea and P{\'e}rez, Aritz},
                  journal={Advances in Neural Information Processing Systems},
                  volume={33},
                  pages={302--312},
                  year={2020}
               }

   - [2] `Mazuelas, S., Shen, Y., & Pérez, A. (2020). Generalized Maximum Entropy for Supervised Classification. arXiv preprint arXiv:2007.05447. <https://arxiv.org/abs/2007.05447>`_

         ::

               @article{mazuelas2020generalized,
                  title={Generalized Maximum Entropy for Supervised Classification},
                  author={Mazuelas, Santiago and Shen, Yuan and P{\'e}rez, Aritz},
                  journal={arXiv preprint arXiv:2007.05447},
                  year={2020}
               }

   - [3] `Bondugula, K., Mazuelas, S., & Pérez, A. (2021). MRCpy: A Library for Minimax Risk Classifiers. arXiv preprint arXiv:2108.01952. <https://arxiv.org/abs/2108.01952>`_

         ::

               @article{bondugula2021mrcpy,
                  title={MRCpy: A Library for Minimax Risk Classifiers},
                  author={Bondugula, Kartheek and Mazuelas, Santiago and P{\'e}rez, Aritz},
                  journal={arXiv preprint arXiv:2108.01952},
                  year={2021}
               }

Funding
^^^^^^^^
Funding in direct support of this work has been provided through different research projects by the following institutions.

.. figure:: fund_logo.png
   :align: center
   :width: 150
   :alt: Spanish Ministry of Science and Innovation logo

   Spanish Ministry of Science and Innovation through the project **PID2019-105058GA-I00** funded by **MCIN/AEI/10.13039/501100011033**.

.. figure:: axalogo.png
   :align: center
   :width: 150
   :alt: AXA Research Fund logo

   AXA Research Fund through the project **"Early Prognosis of COVID-19 Infections via Machine Learning"** funded in the Exceptional Flash Call **"Mitigating risk in the wake of the COVID-19 pandemic"**.

.. figure:: logogobiernovasco.png
   :align: center
   :width: 150
   :alt: Basque Government logo

   Basque Government through the project **"Mathematical Modeling Applied to Health"**, and through the **"ELKARTEK Program"**.

.. |Travis-CI Build Status| image:: https://circleci.com/gh/MachineLearningBCAM/MRCpy.svg?style=shield
   :target: https://circleci.com/gh/MachineLearningBCAM/MRCpy
.. |Code coverage| image:: https://img.shields.io/codecov/c/github/MachineLearningBCAM/MRCpy
   :target: https://codecov.io/gh/MachineLearningBCAM/MRCpy

