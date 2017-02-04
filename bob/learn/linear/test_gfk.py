#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Thu Jun 14 14:45:06 CEST 2012
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test BIC trainer and machine
"""

import bob.learn.linear
import bob.io.matlab
from bob.io.base.test_utils import datafile
from bob.learn.linear import GFKTrainer, GFKMachine
import os


def compute_accuracy(K, Xs, Ys, Xt, Yt):
    import numpy
    numpy.random.seed(10)

    source = numpy.diag(numpy.dot(numpy.dot(Xs, K), Xs.T))
    source = numpy.reshape(source, (Xs.shape[0], 1))
    source = numpy.matlib.repmat(source, 1, Yt.shape[0])

    target = numpy.diag(numpy.dot(numpy.dot(Xt, K), Xt.T))
    target = numpy.reshape(target, (Xt.shape[0], 1))
    target = numpy.matlib.repmat(target, 1, Ys.shape[0]).T

    dist = source + target - 2 * numpy.dot(numpy.dot(Xs, K), Xt.T)

    indices = numpy.argmin(dist, axis=0)
    prediction = Ys[indices]
    accuracy = sum(prediction == Yt)[0] / float(Yt.shape[0])

    return accuracy


def test_matlab_baseline():
    """

    Tests based on this matlab baseline

    http://www-scf.usc.edu/~boqinggo/domainadaptation.html#intro

    """
    import numpy
    numpy.random.seed(10)

    source_webcam = bob.io.matlab.read_matrix(datafile("webcam.mat", __name__))
    webcam_labels = bob.io.matlab.read_matrix(datafile("webcam_labels.mat", __name__))

    target_dslr = bob.io.matlab.read_matrix(datafile("dslr.mat", __name__))
    dslr_labels = bob.io.matlab.read_matrix(datafile("dslr_labels.mat", __name__))

    gfk_trainer = GFKTrainer(10, subspace_dim_source=140, subspace_dim_target=140)
    gfk_machine = gfk_trainer.train(source_webcam, target_dslr)

    accuracy = compute_accuracy(gfk_machine.G, source_webcam, webcam_labels, target_dslr, dslr_labels) * 100
    assert accuracy > 70


def test_trainer():
    """

    Testing the training
    """
    import numpy
    numpy.random.seed(10)

    train_source_data = numpy.random.normal(0, 1, size=(100, 2))
    train_target_data = numpy.random.normal(2, 1, size=(100, 2))

    test_source_data = numpy.random.normal(0, 1, size=(10, 2))
    test_target_data = numpy.random.normal(3, 1, size=(10, 2))

    # Training in random data
    gfk_trainer = GFKTrainer(10,  subspace_dim_source=1.0, subspace_dim_target=1.0)
    gfk_machine = gfk_trainer.train(train_source_data, train_target_data)

    # All the distances are smaller than 1
    products = gfk_machine(test_source_data, test_target_data)
    assert sum(products < 1) == 10

    # Testing the shape
    assert gfk_machine.shape() == (2, 2)

    # Some subspace metrics
    reference = 2.4674011002723324
    assert abs(gfk_machine.compute_principal_angles()-reference) < 0.00001
    assert abs(gfk_machine.compute_binetcouchy_distance() - 0) < 0.00001


def test_trainer_no_norm():
    """

    Testing the training
    """
    import numpy
    numpy.random.seed(10)

    train_source_data = numpy.random.normal(0, 1, size=(100, 2))
    train_target_data = numpy.random.normal(2, 1, size=(100, 2))

    test_source_data = numpy.random.normal(0, 1, size=(10, 2))
    test_target_data = numpy.random.normal(3, 1, size=(10, 2))

    # Training in random data
    gfk_trainer = GFKTrainer(10,  subspace_dim_source=1.0, subspace_dim_target=1.0)
    gfk_machine = gfk_trainer.train(train_source_data, train_target_data, norm_inputs=False)

    # All the distances are smaller than 1
    products = gfk_machine(test_source_data, test_target_data)
    assert sum(products < 1) == 10

    # Testing the shape
    assert gfk_machine.shape() == (2, 2)

    # Some subspace metrics
    reference = 1.6354239731327695
    assert abs(gfk_machine.compute_principal_angles()-reference) < 0.00001
    assert abs(gfk_machine.compute_binetcouchy_distance() - 0) < 0.00001


def test_load_save():
    """

    Testing load and save
    """
    import numpy
    numpy.random.seed(10)

    train_source_data = numpy.random.normal(0, 1, size=(100, 2))
    train_target_data = numpy.random.normal(2, 1, size=(100, 2))

    test_source_data = numpy.random.normal(0, 1, size=(10, 2))
    test_target_data = numpy.random.normal(2, 1, size=(10, 2))

    # Training in random data
    gfk_trainer = GFKTrainer(10,  subspace_dim_source=1.0, subspace_dim_target=1.0)
    gfk_machine = gfk_trainer.train(train_source_data, train_target_data)

    hdf5_file = "gkf.hdf5"
    hdf5 = bob.io.base.HDF5File(hdf5_file, 'w')
    gfk_machine.save(hdf5)
    del gfk_machine
    del hdf5

    hdf5 = bob.io.base.HDF5File(hdf5_file)
    gfk_machine = GFKMachine(hdf5)
    os.remove(hdf5_file)

    # All the distances are smaller than 1
    products = gfk_machine(test_source_data, test_target_data)
    assert sum(products < 1) == 10

    # Testing the shape
    assert gfk_machine.shape() == (2, 2)

    # Some subspace metrics
    reference = 2.4674011002723324
    assert abs(gfk_machine.compute_principal_angles()-reference) < 0.00001
    assert abs(gfk_machine.compute_binetcouchy_distance() - 0) < 0.00001
