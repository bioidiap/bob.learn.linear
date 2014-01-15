#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Tue May 31 16:55:10 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests on the machine infrastructure.
"""

import os, sys
import nose.tools
import math
import numpy

from . import Machine

from xbob.learn.activation import HyperbolicTangent, Identity
from xbob.io import HDF5File

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return __import__('pkg_resources').resource_filename(__name__, os.path.join('data', f))

MACHINE = F('linear-test.hdf5')

def test_initialization():

  # Two inputs and 1 output
  m = Machine(2,1)
  assert (m.weights == 0.0).all()
  nose.tools.eq_( m.weights.shape, (2,1) )
  assert (m.biases == 0.0).all()
  nose.tools.eq_( m.biases.shape, (1,) )

  # Start by providing the data
  w = numpy.array([[0.4, 0.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
  m = Machine(w)
  b = numpy.array([0.3, -3.0], 'float64')
  isub = numpy.array([0., 0.5, 0.5], 'float64')
  idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
  m.input_subtract = isub
  m.input_divide = idiv
  m.biases = b
  m.activation = HyperbolicTangent()

  assert (m.input_subtract == isub).all()
  assert (m.input_divide == idiv).all()
  assert (m.weights == w).all()
  assert (m.biases == b). all()
  nose.tools.eq_(m.activation, HyperbolicTangent())
  # Save to file
  # c = HDF5File("bla.hdf5", 'w')
  # m.save(c)

  # Start by reading data from a file
  c = HDF5File(MACHINE)
  m = Machine(c)
  assert (m.weights == w).all()
  assert (m.biases == b). all()

  # Makes sure we cannot stuff incompatible data
  w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
  m = Machine(w)
  b = numpy.array([0.3, -3.0, 2.7, -18, 52], 'float64') #wrong
  nose.tools.assert_raises(RuntimeError, setattr, m, 'biases', b)
  nose.tools.assert_raises(RuntimeError, setattr, m, 'input_subtract', b)
  nose.tools.assert_raises(RuntimeError, setattr, m, 'input_divide', b)

def test_correctness():

  # Tests the correctness of a linear machine
  c = HDF5File(MACHINE)
  m = Machine(c)

  def presumed(ivalue):
    """Calculates, by hand, the presumed output given the input"""

    # These are the supposed preloaded values from the file "MACHINE"
    isub = numpy.array([0., 0.5, 0.5], 'float64')
    idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
    w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
    b = numpy.array([0.3, -3.0], 'float64')
    act = math.tanh

    return numpy.array([ act((w[i,:]*((ivalue-isub)/idiv)).sum() + b[i]) for i in range(w.shape[0]) ], 'float64')

  testing = [
      [1,1,1],
      [0.5,0.2,200],
      [-27,35.77,0],
      [12,0,0],
      ]

  # 1D case
  maxerr = numpy.ndarray((2,), 'float64')
  maxerr.fill(1e-10)
  for k in testing:
    input = numpy.array(k, 'float64')
    assert (abs(presumed(input) - m(input)) < maxerr).all()

  # 2D case
  output = m(testing)
  for i, k in enumerate(testing):
    input = numpy.array(k, 'float64')
    assert (abs(presumed(input) - output[i,:]) < maxerr).all()

def test_user_allocation():

  # Tests the correctness of a linear machine
  c = HDF5File(MACHINE)
  m = Machine(c)

  def presumed(ivalue):
    """Calculates, by hand, the presumed output given the input"""

    # These are the supposed preloaded values from the file "MACHINE"
    isub = numpy.array([0., 0.5, 0.5], 'float64')
    idiv = numpy.array([0.5, 1.0, 1.0], 'float64')
    w = numpy.array([[0.4, 0.4, 0.2], [0.1, 0.2, 0.7]], 'float64')
    b = numpy.array([0.3, -3.0], 'float64')
    act = math.tanh

    return numpy.array([ act((w[i,:]*((ivalue-isub)/idiv)).sum() + b[i]) for i in range(w.shape[0]) ], 'float64')

  testing = [
      [1,1,1],
      [0.5,0.2,200],
      [-27,35.77,0],
      [12,0,0],
      ]

  # 1D case
  maxerr = numpy.ndarray((2,), 'float64')
  maxerr.fill(1e-10)
  output = numpy.ndarray((2,), 'float64')
  for k in testing:
    input = numpy.array(k, 'float64')
    m(input, output)
    assert (abs(presumed(input) - output) < maxerr).all()

  # 2D case
  output = numpy.ndarray((len(testing), 2), 'float64')
  m(testing, output)
  for i, k in enumerate(testing):
    input = numpy.array(k, 'float64')
    assert (abs(presumed(input) - output[i,:]) < maxerr).all()

def test_comparisons():

  # Start by creating the data
  w1 = numpy.array([[0.4, 0.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
  w2 = numpy.array([[0.4, 1.1], [0.4, 0.2], [0.2, 0.7]], 'float64')
  b1 = numpy.array([0.3, -3.0], 'float64')
  b2 = numpy.array([0.3, 3.0], 'float64')
  isub1 = numpy.array([0., 0.5, 0.5], 'float64')
  isub2 = numpy.array([0.5, 0.5, 0.5], 'float64')
  idiv1 = numpy.array([0.5, 1.0, 1.0], 'float64')
  idiv2 = numpy.array([1.5, 1.0, 1.0], 'float64')

  # Creates Machine's
  m1 = Machine(w1)
  m1.input_subtract = isub1
  m1.input_divide = idiv1
  m1.biases = b1
  m1.activation = HyperbolicTangent()

  m1b = Machine(m1)
  m1c = Machine(w1)
  m1c.input_subtract = isub1
  m1c.input_divide = idiv1
  m1c.biases = b1
  m1c.activation = HyperbolicTangent()

  m2 = Machine(w2)
  m2.input_subtract = isub1
  m2.input_divide = idiv1
  m2.biases = b1
  m2.activation = HyperbolicTangent()

  m3 = Machine(w1)
  m3.input_subtract = isub2
  m3.input_divide = idiv1
  m3.biases = b1
  m3.activation = HyperbolicTangent()

  m4 = Machine(w1)
  m4.input_subtract = isub1
  m4.input_divide = idiv2
  m4.biases = b1
  m4.activation = HyperbolicTangent()

  m5 = Machine(w1)
  m5.input_subtract = isub1
  m5.input_divide = idiv1
  m5.biases = b2
  m5.activation = HyperbolicTangent()

  m6 = Machine(w1)
  m6.input_subtract = isub1
  m6.input_divide = idiv1
  m6.biases = b1
  m6.activation = Identity()

  # Compares them using the overloaded operators == and !=
  assert m1 == m1b
  assert not m1 != m1b
  assert m1.is_similar_to(m1b)
  assert m1 == m1c
  assert not m1 != m1c
  assert m1.is_similar_to(m1c)
  assert not m1 == m2
  assert m1 != m2
  assert not m1.is_similar_to(m2)
  assert not m1 == m3
  assert m1 != m3
  assert not m1.is_similar_to(m3)
  assert not m1 == m4
  assert m1 != m4
  assert not m1.is_similar_to(m4)
  assert not m1 == m5
  assert m1 != m5
  assert not m1.is_similar_to(m5)
  assert not m1 == m6
  assert m1 != m6
  assert not m1.is_similar_to(m6)
