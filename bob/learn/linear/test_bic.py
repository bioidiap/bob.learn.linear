#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Thu Jun 14 14:45:06 CEST 2012
#
# Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

"""Test BIC trainer and machine
"""

import numpy
import nose.tools
import bob.learn.linear

eps = 1e-5

def training_data():
  data = numpy.array([
    [10., 4., 6., 8., 2.],
    [8., 2., 4., 6., 0.],
    [12., 6., 8., 10., 4.],
    [11., 3., 7., 7., 3.],
    [9., 5., 5., 9., 1.]], dtype='float64')

  return data, -1. * data

def eval_data(which):
  eval_data = numpy.ndarray((5,), dtype=numpy.float64)
  if which == 0:
    eval_data.fill(0.)
  elif which == 1:
    eval_data.fill(10.)

  return eval_data

def test_IEC():
  # Tests the IEC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  machine = bob.learn.linear.BICMachine()
  trainer = bob.learn.linear.BICTrainer()

  # train machine with intrapersonal data only
  trainer.train(intra_data, intra_data, machine)
  # => every result should be zero
  assert abs(machine(eval_data(0))) < eps
  assert abs(machine(eval_data(1))) < eps

  # re-train the machine with intra- and extrapersonal data
  trainer.train(intra_data, extra_data, machine)
  # now, only the input vector 0 should give log-likelihood 0
  assert abs(machine(eval_data(0))) < eps
  # while a positive vector should give a positive result
  assert machine(eval_data(1)) > 0.

  machine2 = trainer.train(intra_data, extra_data)
  assert machine.is_similar_to(machine2)

@nose.tools.raises(RuntimeError)
def test_raises():

  # Tests the BIC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  trainer = bob.learn.linear.BICTrainer(2,2)

  # The data are chosen such that the third eigenvalue is zero.
  # Hence, calculating rho (i.e., using the Distance From Feature Space) is impossible
  machine = bob.learn.linear.BICMachine(True)
  trainer.train(intra_data, intra_data, machine)

def test_BIC():
  # Tests the BIC training of the BICTrainer
  intra_data, extra_data = training_data()

  # train BIC machine
  trainer = bob.learn.linear.BICTrainer(2,2)

  # So, now without rho...
  machine = bob.learn.linear.BICMachine(False)

  # First, train the machine with intrapersonal data only
  trainer.train(intra_data, intra_data, machine)

  assert machine.input_size == 5

  # => every result should be zero
  assert abs(machine(eval_data(0))) < eps
  assert abs(machine(eval_data(1))) < eps

  # re-train the machine with intra- and extrapersonal data
  trainer.train(intra_data, extra_data, machine)
  # now, only the input vector 0 should give log-likelihood 0
  assert abs(machine(eval_data(0))) < eps
  # while a positive vector should give a positive result
  assert machine(eval_data(1)) > 0.

  machine2 = trainer.train(intra_data, extra_data)
  # For some reason, the == test fails on 32 bit machines
#  assert machine == machine2
  # But, in fact the machines should be identical.
  assert machine.is_similar_to(machine2, 1e-10, 1e-15)

def test_bic_split():
  # Tests the auxiliary function bic_intra_extra_pairs
  data = [[1,2,3],[4,5,6],[7,8,9]]
  intra_pairs, extra_pairs = bob.learn.linear.bic_intra_extra_pairs(data)

  # check number of pairs
  assert len(intra_pairs) == 9
  assert len(extra_pairs) == 27

  # check exact intra pairs
  for c in data:
    for v1 in c:
      for v2 in c:
        if v1 != v2:
          # check that exactly one of the two possible pairs are inside
          assert ((v1,v2) in intra_pairs) != ((v2,v1) in intra_pairs)

  # check extra_pairs
  for c1 in data:
    for c2 in data:
      if c1 != c2:
        for v1 in c1:
          for v2 in c2:
            # check that exactly one of the two possible pairs is inside
            assert ((v1,v2) in extra_pairs) != ((v2,v1) in extra_pairs)

def test_bic_split_between_factors():
  # Tests the auxiliary function bic_intra_extra_pairs_between_factors
  factor1 = [[1,2,3],[4,5,6],[7,8,9]]
  factor2 = [[11,12,13],[14,15,16],[17,18,19]]
  intra_pairs, extra_pairs = bob.learn.linear.bic_intra_extra_pairs_between_factors(factor1, factor2)

  # check number of pairs
  assert len(intra_pairs) == 27
  assert len(extra_pairs) == 54

  # assert that all pairs are taken from factor 1 and factor 2, in the right order
  assert all(p[0] < 10 and p[1] > 10 for pairs in (intra_pairs,extra_pairs) for p in pairs)

  # check intra pairs
  for c1, c2 in zip(factor1, factor2):
    for f1 in c1:
      for f2 in c2:
        assert (f1, f2) in intra_pairs

  # check extra pairs
  for i1 in range(3):
    for i2 in range(3):
      if i1 != i2:
        for f1 in factor1[i1]:
          for f2 in factor2[i2]:
            assert (f1, f2) in extra_pairs
