/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 14 Jan 2014 14:26:09 CET
 *
 * @brief Bindings for a LinearMachine
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#define XBOB_LEARN_LINEAR_MODULE
#include "cleanup.h"
#include <xbob.blitz/cppapi.h>
#include <xbob.io/api.h>
#include <xbob.learn.activation/api.h>
#include <xbob.learn.linear/api.h>
#include <structmember.h>

/**********************************************
 * Implementation of LinearMachine base class *
 **********************************************/

PyDoc_STRVAR(s_linear_str, XBOB_EXT_MODULE_PREFIX ".Machine");

PyDoc_STRVAR(s_linear_doc,
"Machine([input_size=0, [output_size=0]])\n\
Machine(weights)\n\
Machine(config)\n\
Machine(other)\n\
\n\
A linear classifier. See C. M. Bishop, 'Pattern Recognition\n\
and Machine  Learning', chapter 4 for more details. The basic\n\
matrix operation performed for projecting the input to the\n\
output is: :math:`o = w \\times i` (with :math:`w` being the\n\
vector of machine weights and :math:`i` the input data vector).\n\
The weights matrix is therefore organized column-wise. In this\n\
scheme, each column of the weights matrix can be interpreted\n\
as vector to which the input is projected. The number of\n\
columns of the weights matrix determines the number of outputs\n\
this linear machine will have. The number of rows, the number\n\
of allowed inputs it can process.\n\
\n\
Input and output is always performed on 1D arrays with 64-bit\n\
floating point numbers.\n\
\n\
A linear machine can be constructed in different ways. In the\n\
first form, the user specifies optional input and output vector\n\
sizes. The machine is remains **uninitialized**. With the second\n\
form, the user passes a 2D array with 64-bit floats containing\n\
weight matrix to be used by the new machine. In the third form\n\
the user passes a pre-opened HDF5 file pointing to the machine\n\
information to be loaded in memory. Finally, in the last form\n\
(copy constructor), the user passes another\n\
:py:class:`Machine` that will be fully copied.\n\
");

static int PyBobLearnLinearMachine_init_sizes(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"input_size", "output_size", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t input_size = 0;
  Py_ssize_t output_size = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|nn", kwlist,
        &input_size, &output_size)) return -1;

  try {
    self->machine = new bob::machine::LinearMachine(input_size, output_size);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", s_linear_str);
  }

  return 0;

}

static int PyBobLearnLinearMachine_init_weights(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"weights", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* weights = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        &PyBlitzArray_Converter, &weights)) return -1;
  auto weights_ = make_safe(weights);

  if (weights->type_num != NPY_FLOAT64 || weights->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "LinearMachine only supports 64-bit floats 2D arrays for property array `weights'");
    return -1;
  }

  try {
    self->machine = new bob::machine::LinearMachine
      (*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", s_linear_str);
  }

  return 0;

}

static int PyBobLearnLinearMachine_init_hdf5(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"config", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* config = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
        &config)) return -1;

  if (!PyBobIoHDF5File_Check(config)) {
    PyErr_Format(PyExc_TypeError, "initialization with HDF5 files requires an object of type `HDF5File' for input, not `%s'", config->ob_type->tp_name);
    return -1;
  }

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(config);

  try {
    self->machine = new bob::machine::LinearMachine(*(h5f->f));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", s_linear_str);
  }

  return 0;

}

static int PyBobLearnLinearMachine_init_copy(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist,
        &other)) return -1;

  if (!PyBobLearnLinearMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "copy construction requires an object of type `%s' for input, not `%s'", self->ob_type->tp_name, other->ob_type->tp_name);
    return -1;
  }

  auto copy = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);

  try {
    self->machine = new bob::machine::LinearMachine(*(copy->machine));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", s_linear_str);
  }

  return 0;

}

static int PyBobLearnLinearMachine_init(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwds?PyDict_Size(kwds):0;

  switch (nargs) {

    case 0: //default initializer
    case 2: //two sizes
      return PyBobLearnLinearMachine_init_sizes(self, args, kwds);

    case 1:

      {

        PyObject* arg = 0; ///< borrowed (don't delete)
        if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
        else {
          PyObject* tmp = PyDict_Values(kwds);
          auto tmp_ = make_safe(tmp);
          arg = PyList_GET_ITEM(tmp, 0);
        }

        if (PyBobIoHDF5File_Check(arg)) {
          return PyBobLearnLinearMachine_init_hdf5(self, args, kwds);
        }

        if (PyBlitzArray_Check(arg) || PyArray_Check(arg)) {
          return PyBobLearnLinearMachine_init_weights(self, args, kwds);
        }

        if (PyNumber_Check(arg)) {
          return PyBobLearnLinearMachine_init_sizes(self, args, kwds);
        }

        if (PyBobMachineActivation_Check(arg)) {
          return PyBobLearnLinearMachine_init_copy(self, args, kwds);
        }

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", s_linear_str, arg->ob_type->tp_name);

      }

      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", s_linear_str, nargs);

  }

  return -1;

}

static void PyBobLearnLinearMachine_delete (PyBobLearnLinearMachineObject* self) {

  delete self->machine;
  self->ob_type->tp_free((PyObject*)self);

}

int PyBobLearnLinearMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearMachine_Type));
}

static PyObject* PyBobLearnLinearMachine_RichCompare (PyBobLearnLinearMachineObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        s_linear_str, other->ob_type->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);

  switch (op) {
    case Py_EQ:
      if (self->machine->operator==(*other_->machine)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (self->machine->operator!=(*other_->machine)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

static PyMethodDef PyBobLearnLinearMachine_methods[] = {
  {0} /* Sentinel */
};

/**
    .add_property("activation", &bob::machine::LinearMachine::getActivation, &bob::machine::LinearMachine::setActivation, "The activation function - by default, the identity function. The output provided by the activation function is passed, unchanged, to the user.")
**/

PyDoc_STRVAR(s_weights_str, "weights");
PyDoc_STRVAR(s_weights_doc,
"Weight matrix to which the input is projected to. The output\n\
of the project is fed subject to bias and activation before\n\
being output.\n\
");

static PyObject* PyBobLearnLinearMachine_getWeights
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->machine->getWeights()));
}

static int PyBobLearnLinearMachine_setWeights (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* weights = 0;
  if (!PyBlitzArray_Converter(o, &weights)) return -1;
  auto weights_ = make_safe(weights);

  if (weights->type_num != NPY_FLOAT64 || weights->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "LinearMachine only supports 64-bit floats 2D arrays for property array `weights'");
    return -1;
  }

  try {
    self->machine->setWeights(*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `weights' of %s: unknown exception caught", s_linear_str);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_biases_str, "biases");
PyDoc_STRVAR(s_biases_doc,
"Bias to the output units of this linear machine, to be added\n\
to the output before activation.\n\
");

static PyObject* PyBobLearnLinearMachine_getBiases
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->machine->getBiases()));
}

static int PyBobLearnLinearMachine_setBiases (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* biases = 0;
  if (!PyBlitzArray_Converter(o, &biases)) return -1;
  auto biases_ = make_safe(biases);

  if (biases->type_num != NPY_FLOAT64 || biases->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "LinearMachine only supports 64-bit floats 1D arrays for property array `biases'");
    return -1;
  }

  try {
    self->machine->setBiases(*PyBlitzArrayCxx_AsBlitz<double,1>(biases));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `biases' of %s: unknown exception caught", s_linear_str);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_input_subtract_str, "input_subtract");
PyDoc_STRVAR(s_input_subtract_doc,
"Input subtraction factor, before feeding data through the\n\
weight matrix W. The subtraction is the first applied\n\
operation in the processing chain - by default, it is set to\n\
0.0.\n\
");

static PyObject* PyBobLearnLinearMachine_getInputSubtraction
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->machine->getInputSubtraction()));
}

static int PyBobLearnLinearMachine_setInputSubtraction
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* input_subtract = 0;
  if (!PyBlitzArray_Converter(o, &input_subtract)) return -1;
  auto input_subtract_ = make_safe(input_subtract);

  if (input_subtract->type_num != NPY_FLOAT64 || input_subtract->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "LinearMachine only supports 64-bit floats 1D arrays for property array `input_subtract'");
    return -1;
  }

  try {
    self->machine->setInputSubtraction(*PyBlitzArrayCxx_AsBlitz<double,1>(input_subtract));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `input_subtract' of %s: unknown exception caught", s_linear_str);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_input_divide_str, "input_divide");
PyDoc_STRVAR(s_input_divide_doc,
"Input division factor, before feeding data through the\n\
weight matrix W. The division is applied just after\n\
subtraction - by default, it is set to 1.0.\n\
");

static PyObject* PyBobLearnLinearMachine_getInputDivision
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->machine->getInputDivision()));
}

static int PyBobLearnLinearMachine_setInputDivision (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* input_divide = 0;
  if (!PyBlitzArray_Converter(o, &input_divide)) return -1;
  auto input_divide_ = make_safe(input_divide);

  if (input_divide->type_num != NPY_FLOAT64 || input_divide->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "LinearMachine only supports 64-bit floats 1D arrays for property array `input_divide'");
    return -1;
  }

  try {
    self->machine->setInputDivision(*PyBlitzArrayCxx_AsBlitz<double,1>(input_divide));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `input_divide' of %s: unknown exception caught", s_linear_str);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_shape_str, "shape");
PyDoc_STRVAR(s_shape_doc,
"A tuple that represents the size of the input vector\n\
followed by the size of the output vector in the format\n\
``(input, output)``.\n\
");

static PyObject* PyBobLearnLinearMachine_getShape
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return Py_BuildValue("(nn)", self->machine->inputSize(),
      self->machine->outputSize());
}

static int PyBobLearnLinearMachine_setShape (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  if (!PySequence_Check(o)) {
    PyErr_Format(PyExc_TypeError, "LinearMachine shape can only be set using tuples (or sequences), not `%s'", o->ob_type->tp_name);
    return -1;
  }

  PyObject* shape = PySequence_Tuple(o);
  auto shape_ = make_safe(shape);

  if (PyTuple_GET_SIZE(shape) != 2) {
    PyErr_Format(PyExc_RuntimeError, "LinearMachine shape can only be set using  2-position tuples (or sequences), not an %" PY_FORMAT_SIZE_T "d-position sequence", PyTuple_GET_SIZE(shape));
    return -1;
  }

  Py_ssize_t in = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 0), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;
  Py_ssize_t out = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 1), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  try {
    self->machine->resize(in, out);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `shape' of %s: unknown exception caught", s_linear_str);
    return -1;
  }

  return 0;

}

static PyGetSetDef PyBobLearnLinearMachine_getseters[] = {
    {
      s_weights_str,
      (getter)PyBobLearnLinearMachine_getWeights,
      (setter)PyBobLearnLinearMachine_setWeights,
      s_weights_doc,
      0
    },
    {
      s_biases_str,
      (getter)PyBobLearnLinearMachine_getBiases,
      (setter)PyBobLearnLinearMachine_setBiases,
      s_biases_doc,
      0
    },
    {
      s_input_subtract_str,
      (getter)PyBobLearnLinearMachine_getInputSubtraction,
      (setter)PyBobLearnLinearMachine_setInputSubtraction,
      s_input_subtract_doc,
      0
    },
    {
      s_input_divide_str,
      (getter)PyBobLearnLinearMachine_getInputDivision,
      (setter)PyBobLearnLinearMachine_setInputDivision,
      s_input_divide_doc,
      0
    },
    {
      s_shape_str,
      (getter)PyBobLearnLinearMachine_getShape,
      (setter)PyBobLearnLinearMachine_setShape,
      s_shape_doc,
      0
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearMachine_Type = {
    PyObject_HEAD_INIT(0)
    0,                                                /* ob_size */
    s_linear_str,                                     /* tp_name */
    sizeof(PyBobLearnLinearMachineObject),            /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearMachine_delete,       /* tp_dealloc */
    0,                                                /* tp_print */
    0,                                                /* tp_getattr */
    0,                                                /* tp_setattr */
    0,                                                /* tp_compare */
    0,                                                /* tp_repr */
    0,                                                /* tp_as_number */
    0,                                                /* tp_as_sequence */
    0,                                                /* tp_as_mapping */
    0,                                                /* tp_hash */
    0, //(ternaryfunc)PyBobLearnLinearMachine_call,        /* tp_call */
    0,                                                /* tp_str */
    0,                                                /* tp_getattro */
    0,                                                /* tp_setattro */
    0,                                                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,         /* tp_flags */
    s_linear_doc,                                     /* tp_doc */
    0,                                                /* tp_traverse */
    0,                                                /* tp_clear */
    (richcmpfunc)PyBobLearnLinearMachine_RichCompare, /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    PyBobLearnLinearMachine_methods,                  /* tp_methods */
    0,                                                /* tp_members */
    PyBobLearnLinearMachine_getseters,                /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)PyBobLearnLinearMachine_init,           /* tp_init */
    0,                                                /* tp_alloc */
    0,                                                /* tp_new */
};

/******
static object forward(const bob::machine::LinearMachine& m,
  bob::python::const_ndarray input)
{
  const bob::core::array::typeinfo& info = input.type();

  switch(info.nd) {
    case 1:
      {
        bob::python::ndarray output(bob::core::array::t_float64, m.outputSize());
        blitz::Array<double,1> output_ = output.bz<double,1>();
        m.forward(input.bz<double,1>(), output_);
        return output.self();
      }
    case 2:
      {
        bob::python::ndarray output(bob::core::array::t_float64, info.shape[0], m.outputSize());
        blitz::Array<double,2> input_ = input.bz<double,2>();
        blitz::Array<double,2> output_ = output.bz<double,2>();
        blitz::Range all = blitz::Range::all();
        for (size_t k=0; k<info.shape[0]; ++k) {
          blitz::Array<double,1> i_ = input_(k,all);
          blitz::Array<double,1> o_ = output_(k,all);
          m.forward(i_, o_);
        }
        return output.self();
      }
    default:
      PYTHON_ERROR(TypeError, "cannot forward arrays with "  SIZE_T_FMT " dimensions (only with 1 or 2 dimensions).", info.nd);
  }
}

static void forward2(const bob::machine::LinearMachine& m,
    bob::python::const_ndarray input, bob::python::ndarray output)
{
  const bob::core::array::typeinfo& info = input.type();

  switch(info.nd) {
    case 1:
      {
        blitz::Array<double,1> output_ = output.bz<double,1>();
        m.forward(input.bz<double,1>(), output_);
      }
      break;
    case 2:
      {
        blitz::Array<double,2> input_ = input.bz<double,2>();
        blitz::Array<double,2> output_ = output.bz<double,2>();
        blitz::Range all = blitz::Range::all();
        for (size_t k=0; k<info.shape[0]; ++k) {
          blitz::Array<double,1> i_ = input_(k,all);
          blitz::Array<double,1> o_ = output_(k,all);
          m.forward(i_, o_);
        }
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "cannot forward arrays with "  SIZE_T_FMT " dimensions (only with 1 or 2 dimensions).", info.nd);
  }
}
***/

/***
void bind_machine_linear() {
    .def("is_similar_to", &bob::machine::LinearMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this LinearMachine with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::LinearMachine::load, (arg("self"), arg("config")), "Loads the weights and biases from a configuration file. Both weights and biases have their dimensionalities checked between each other for consistency.")
    .def("save", &bob::machine::LinearMachine::save, (arg("self"), arg("config")), "Saves the weights and biases to a configuration file.")
    .def("resize", &bob::machine::LinearMachine::resize, (arg("self"), arg("input"), arg("output")), "Resizes the machine. If either the input or output increases in size, the weights and other factors should be considered uninitialized. If the size is preserved or reduced, already initialized values will not be changed.\n\nTip: Use this method to force data compression. All will work out given most relevant factors to be preserved are organized on the top of the weight matrix. In this way, reducing the system size will supress less relevant projections.")
    .def("__call__", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("forward", &forward2, (arg("self"), arg("input"), arg("output")), "Projects the input to the weights and biases and saves results on the output")
    .def("__call__", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward, (arg("self"), arg("input")), "Projects the input to the weights and biases and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}
***/
