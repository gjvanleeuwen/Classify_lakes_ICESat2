.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/ICESat_lake_classification.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/ICESat_lake_classification
    .. image:: https://readthedocs.org/projects/ICESat_lake_classification/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://ICESat_lake_classification.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/ICESat_lake_classification/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/ICESat_lake_classification
    .. image:: https://img.shields.io/pypi/v/ICESat_lake_classification.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/ICESat_lake_classification/
    .. image:: https://img.shields.io/conda/vn/conda-forge/ICESat_lake_classification.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/ICESat_lake_classification
    .. image:: https://pepy.tech/badge/ICESat_lake_classification/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/ICESat_lake_classification
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/ICESat_lake_classification

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========================
ICESat_lake_classification
==========================


    Project to derive unsupervised lake classification for ICESat-2 ATL03 data

ICESat-2 is a novel Satellite mission carrying a LIDAR instrument which delivers photon heights over the arctic and antarctic latitudes. The ATL03 photon cloud data can indicate clear surfaces over smooth terrain like oce and water. The photon cloud may have second reflections from the bottom of a lake which can be leveraged to classify lakes. These lakes maybe located on Glaciers, which is the interest of this project.

This library tries using novel machine learning techniques to classify lakes over glaciers using the aforementioned ATL03 ICESat-2 data.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
