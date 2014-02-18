.. vim: set fileencoding=utf-8 :
.. Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
.. Wed Mar 14 12:31:35 2012 +0100
.. modified by Elie Khoury <elie.khoury@idiap.ch>
.. Mon May 06 15:50:20 2013 +0100
.. consolidated by Andre Anjos <andre.anjos@idiap.ch>
.. Wed 15 Jan 2014 12:20:47 CET
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. testsetup:: *

   import os
   import numpy
   import tempfile
   import xbob.learn.linear

   current_directory = os.path.realpath(os.curdir)
   temp_dir = tempfile.mkdtemp(prefix='bob_doctest_')
   os.chdir(temp_dir)

==============================
 Linear Machines and Trainers
==============================

Machines are one of the core components of |project|. They represent
statistical models or other functions defined by parameters that can be learnt
or manually set. The simplest of |project|'s machines is a
:py:class:`xbob.learn.linear.Machine`. This package contains the definition of
this class as well as trainers that can learn linear machine parameters from
data.

Linear machines
---------------

Linear machines execute the simple operation :math:`y = \mathbf{W} x`, where
:math:`y` is the output vector, :math:`x` is the input vector and :math:`W` is
a matrix (2D array) stored in the machine. The input vector :math:`x` should be
composed of double-precision floating-point elements. The output will also be
in double-precision. Here is how to use a
:py:class:`xbob.learn.linear.Machine`:

.. doctest::

  >>> W = numpy.array([[0.5, 0.5], [1.0, 1.0]], 'float64')
  >>> W
  array([[ 0.5,  0.5],
         [ 1. ,  1. ]])
  >>> machine = xbob.learn.linear.Machine(W)
  >>> machine.shape
  (2, 2)
  >>> x = numpy.array([0.3, 0.4], 'float64')
  >>> y = machine(x)
  >>> y
  array([ 0.55,  0.55])

As was shown in the above example, the way to pass data through a machine is to
call its :py:meth:`xbob.learn.linear.Machine.__call__()` operator.

The first thing to notice about machines is that they can be stored and
retrieved in HDF5 files (for more details in manipulating HDF5 files, please
consult the documentation for :py:mod:`xbob.io`. To save the before metioned
machine to a file, just use the machine's
:py:meth:`xbob.learn.linear.Machine.save`` command. Because several machines
can be stored on the same :py:class:`xbob.io.HDF5File`, we let the user open
the file and set it up before the machine can write to it:

.. doctest::

  >>> myh5_file = xbob.io.HDF5File('linear.hdf5', 'w')
  >>> #do other operations on myh5_file to set it up, optionally
  >>> machine.save(myh5_file)
  >>> del myh5_file #close

You can load the machine again in a similar way:

.. doctest::

  >>> myh5_file = xbob.io.HDF5File('linear.hdf5')
  >>> reloaded = xbob.learn.linear.Machine(myh5_file)
  >>> numpy.array_equal(machine.weights, reloaded.weights)
  True

The shape of a :py:class:`xbob.learn.linear.Machine` (see
:py:attr:`xbob.learn.linear.Machine.shape`) indicates the size of the input
vector that is expected by this machine and the size of the output vector it
produces, in a tuple format like ``(input_size, output_size)``:

.. doctest::

  >>> machine.shape
  (2, 2)

A :py:class:`xbob.learn.linear.Machine`` also supports pre-setting
normalization vectors that are applied to every input :math:`x`. You can set a
subtraction factor and a division factor, so that the actual input :math:`x'`
that is fed to the matrix :math:`W` is :math:`x' = (x - s) ./ d`. The variables
:math:`s` and :math:`d` are vectors that have to have the same size as the
input vector :math:`x`. The operator :math:`./` indicates an element-wise
division. By default, :math:`s := 0.0` and :math:`d := 1.0`.

.. doctest::

  >>> machine.input_subtract
  array([ 0.,  0.])
  >>> machine.input_divide
  array([ 1.,  1.])

To set a new value for :math:`s` or :math:`d` just assign the desired machine
property:

.. doctest::

  >>> machine.input_subtract = numpy.array([0.5, 0.8])
  >>> machine.input_divide = numpy.array([2.0, 4.0])
  >>> y = machine(x)
  >>> y
  array([-0.15, -0.15])

.. note::

  In the event you save a machine that has the subtraction and/or a division
  factor set, the vectors are saved and restored automatically w/o user
  intervention.

Linear machine trainers
-----------------------

Next, we examine available ways to train a :py:class:`xbob.learn.linear.Machine`
so they can do something useful for you.

Principal component analysis
============================

**PCA** [1]_ is one way to train a :py:class:`xbob.learn.linear.Machine`. The
associated |project| class is :py:class:`xbob.learn.linear.PCATrainer` as the
training procedure mainly relies on a singular value decomposition.

**PCA** belongs to the category of `unsupervised` learning algorithms, which
means that the training data is not labelled. Therefore, the training set can
be represented by a set of features stored in a container. Using |project|,
this container is a 2D :py:class:`numpy.ndarray`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> data = numpy.array([[3,-3,100], [4,-4,50], [3.5,-3.5,-50], [3.8,-3.7,-100]], dtype='float64')
   >>> print(data)
   [[   3.    -3.   100. ]
    [   4.    -4.    50. ]
    [   3.5   -3.5  -50. ]
    [   3.8   -3.7 -100. ]]

Once the training set has been defined, the overall procedure to train a
:py:class:`xbob.learn.linear.Machine` with a
:py:class:`xbob.learn.linear.PCATrainer` is simple and shown below. Please note
that the concepts remains very similar for most of the other `trainers` and
`machines`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = xbob.learn.linear.PCATrainer() # Creates a PCA trainer
   >>> [machine, eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print(machine.weights)  # The weights of the returned (linear) Machine after the training procedure
   [[ 0.002 -0.706 -0.708]
    [-0.002  0.708 -0.706]
    [-1.    -0.003 -0.   ]]

Next, input data can be projected using this learned projection matrix
:math:`W`.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> e = numpy.array([3.2,-3.3,-10], 'float64')
   >>> print(machine(e))
   [ 9.999 0.47 0.092]


Linear discriminant analysis
============================

**LDA** [2]_ is another way to train a :py:class:`xbob.learn.linear.Machine`.
The associated |project| class is
:py:class:`xbob.learn.linear.FisherLDATrainer`.

In contrast to **PCA** [1]_, **LDA** [2]_ is a `supervised` technique.
Furthermore, the training data should be organized differently. It is indeed
required to be a list of 2D :py:class:`numpy.ndarray`\'s, one for each class.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> data1 = numpy.array([[3,-3,100], [4,-4,50], [40,-40,150]], dtype='float64')
   >>> data2 = numpy.array([[3,6,-50], [4,8,-100], [40,79,-800]], dtype='float64')
   >>> data = [data1,data2]

Once the training set has been defined, the procedure to train the
:py:class:`xbob.learn.linear.Machine` with **LDA** is very similar to the one
for **PCA**. This is shown below.

.. doctest::
   :options: +NORMALIZE_WHITESPACE

   >>> trainer = xbob.learn.linear.FisherLDATrainer()
   >>> [machine,eig_vals] = trainer.train(data)  # Trains the machine with the given data
   >>> print(eig_vals)  # doctest: +SKIP
   [ 13.10097786 0. ]
   >>> machine.resize(3,1)  # Make the output space of dimension 1
   >>> print(machine.weights)  # The new weights after the training procedure
   [[ 0.609]
    [ 0.785]
    [ 0.111]]


.. Place here your external references
.. [1] http://en.wikipedia.org/wiki/Principal_component_analysis
.. [2] http://en.wikipedia.org/wiki/Linear_discriminant_analysis
