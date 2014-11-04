.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 13 Dec 2013 12:50:06 CET
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. _bob.learn.linear:

==================================
 Bob Linear Machines and Trainers
==================================

.. todolist::

This package includes the definition of a linear machine, which is capable of
either projecting the input data into maximally spread representations,
linearly, or providing linear separation planes for multi-class data samples.
The package includes the machine definition *per se* and a selection of
different trainers for specialized purposes:

 * Principal Component Analysis
 * Fisher's Linear Discriminant Analysis
 * (Conjugate Gradient) Logistic Regression
 * Whitening filter
 * Within-class covariance normalization (WCCN)

Documentation
-------------

.. toctree::
   :maxdepth: 2

   guide
   py_api
   c_cpp_api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
