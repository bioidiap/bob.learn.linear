#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines.utils import assert_picklable
from bob.learn.linear import Machine
import numpy
import pickle


def test_machine():
    machine = Machine(10,3) 
    machine.weights = numpy.arange(30).reshape(10,3).astype("float")
    machine.input_div = numpy.arange(3).astype("float")
    machine.input_sub = numpy.arange(3).astype("float")

    machine_after_pickle = pickle.loads(pickle.dumps(machine))
    
    assert numpy.allclose(machine.weights, machine_after_pickle.weights, 10e-3)
    assert numpy.allclose(machine.input_div, machine_after_pickle.input_div, 10e-3)
    assert numpy.allclose(machine.input_sub, machine_after_pickle.input_sub, 10e-3)