.. analogs finder master file, created by
   sphinx-quickstart on Mon Dec  4 11:58:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cluster Algorithms for Drug Discovery
===========================================


Github : https://github.com/danielSoler93/cluster_drug_discovery

Cluster drug discovery is a specific python package to
handle clustering in drug discovery. The main aim of
the package is not to give a general cluster method that
works in all cases (as it does not exist) but to give the
possibility to clusterize in an easy manner and to asses
the quality of the results. 



.. figure:: images/cluster.png
    :scale: 80%
    :align: center


Structure of the Package
==========================

The package consist of three parts:

1) An input preprocessing to rapidly convert features from your pdb, xtc, dcd to numerical values

2) The Clustering main class containing a high level object to wrap all cluster algorithms

3) An analysis package to asses how good is your cluster and what could be improved


.. figure:: images/structure.png
    :scale: 80%
    :align: center


Documentation
===================

.. toctree::
   installation/index.rst


.. toctree::
   tutorial/index.rst


.. toctree::
   changelog/index.rst

