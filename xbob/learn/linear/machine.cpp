/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 14 Jan 2014 14:26:09 CET
 *
 * @brief Bindings for a LinearMachine
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#define XBOB_LEARN_LINEAR_MODULE
#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>
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

static int PyBobLearnLinearMachine_init_sizes
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"input_size", "output_size", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t input_size = 0;
  Py_ssize_t output_size = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|nn", kwlist,
        &input_size, &output_size)) return -1;

  try {
    self->cxx = new bob::machine::LinearMachine(input_size, output_size);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
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
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 2D arrays for property array `weights'", Py_TYPE(self)->tp_name);
    return -1;
  }

  try {
    self->cxx = new bob::machine::LinearMachine
      (*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static int PyBobLearnLinearMachine_init_hdf5(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"config", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* config = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobIoHDF5File_Type, &config)) return -1;

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(config);

  try {
    self->cxx = new bob::machine::LinearMachine(*(h5f->f));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static int PyBobLearnLinearMachine_init_copy
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearMachine_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);

  try {
    self->cxx = new bob::machine::LinearMachine(*(copy->cxx));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static int PyBobLearnLinearMachine_init(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

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

        if (PyBobLearnLinearMachine_Check(arg)) {
          return PyBobLearnLinearMachine_init_copy(self, args, kwds);
        }

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);

      }

      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0, 1 or 2 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);

  }

  return -1;

}

static void PyBobLearnLinearMachine_delete
(PyBobLearnLinearMachineObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);

}

int PyBobLearnLinearMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearMachine_Type));
}

static PyObject* PyBobLearnLinearMachine_RichCompare
(PyBobLearnLinearMachineObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);

  switch (op) {
    case Py_EQ:
      if (self->cxx->operator==(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (self->cxx->operator!=(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

PyDoc_STRVAR(s_weights_str, "weights");
PyDoc_STRVAR(s_weights_doc,
"Weight matrix to which the input is projected to. The output\n\
of the project is fed subject to bias and activation before\n\
being output.\n\
");

static PyObject* PyBobLearnLinearMachine_getWeights
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getWeights()));
}

static int PyBobLearnLinearMachine_setWeights (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* weights = 0;
  if (!PyBlitzArray_Converter(o, &weights)) return -1;
  auto weights_ = make_safe(weights);

  if (weights->type_num != NPY_FLOAT64 || weights->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 2D arrays for property array `weights'", Py_TYPE(self)->tp_name);
    return -1;
  }

  try {
    self->cxx->setWeights(*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `weights' of %s: unknown exception caught", Py_TYPE(self)->tp_name);
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
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getBiases()));
}

static int PyBobLearnLinearMachine_setBiases (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* biases = 0;
  if (!PyBlitzArray_Converter(o, &biases)) return -1;
  auto biases_ = make_safe(biases);

  if (biases->type_num != NPY_FLOAT64 || biases->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `biases'", Py_TYPE(self)->tp_name);
    return -1;
  }

  try {
    self->cxx->setBiases(*PyBlitzArrayCxx_AsBlitz<double,1>(biases));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `biases' of %s: unknown exception caught", Py_TYPE(self)->tp_name);
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
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getInputSubtraction()));
}

static int PyBobLearnLinearMachine_setInputSubtraction
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* input_subtract = 0;
  if (!PyBlitzArray_Converter(o, &input_subtract)) return -1;
  auto input_subtract_ = make_safe(input_subtract);

  if (input_subtract->type_num != NPY_FLOAT64 || input_subtract->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `input_subtract'", Py_TYPE(self)->tp_name);
    return -1;
  }

  try {
    self->cxx->setInputSubtraction(*PyBlitzArrayCxx_AsBlitz<double,1>(input_subtract));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `input_subtract' of %s: unknown exception caught", Py_TYPE(self)->tp_name);
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
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getInputDivision()));
}

static int PyBobLearnLinearMachine_setInputDivision (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {

  PyBlitzArrayObject* input_divide = 0;
  if (!PyBlitzArray_Converter(o, &input_divide)) return -1;
  auto input_divide_ = make_safe(input_divide);

  if (input_divide->type_num != NPY_FLOAT64 || input_divide->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `input_divide'", Py_TYPE(self)->tp_name);
    return -1;
  }

  try {
    self->cxx->setInputDivision(*PyBlitzArrayCxx_AsBlitz<double,1>(input_divide));
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `input_divide' of %s: unknown exception caught", Py_TYPE(self)->tp_name);
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
  return Py_BuildValue("(nn)", self->cxx->inputSize(),
      self->cxx->outputSize());
}

static int PyBobLearnLinearMachine_setShape
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {

  if (!PySequence_Check(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' shape can only be set using tuples (or sequences), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  PyObject* shape = PySequence_Tuple(o);
  auto shape_ = make_safe(shape);

  if (PyTuple_GET_SIZE(shape) != 2) {
    PyErr_Format(PyExc_RuntimeError, "`%s' shape can only be set using  2-position tuples (or sequences), not an %" PY_FORMAT_SIZE_T "d-position sequence", Py_TYPE(self)->tp_name, PyTuple_GET_SIZE(shape));
    return -1;
  }

  Py_ssize_t in = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 0), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;
  Py_ssize_t out = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 1), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  try {
    self->cxx->resize(in, out);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `shape' of %s: unknown exception caught", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_activation_str, "activation");
PyDoc_STRVAR(s_activation_doc,
"The activation function - by default, the identity function.\n\
The output provided by the activation function is passed,\n\
unchanged, to the user.\n\
");

static PyObject* PyBobLearnLinearMachine_getActivation
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
  return PyBobLearnActivation_NewFromActivation(self->cxx->getActivation());
}

static int PyBobLearnLinearMachine_setActivation
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {

  if (!PyBobLearnActivation_Check(o)) {
    PyErr_Format(PyExc_TypeError, "%s activation requires an object of type `Activation' (or an inherited type), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  auto py = reinterpret_cast<PyBobLearnActivationObject*>(o);
  self->cxx->setActivation(py->cxx);
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
    {
      s_activation_str,
      (getter)PyBobLearnLinearMachine_getActivation,
      (setter)PyBobLearnLinearMachine_setActivation,
      s_activation_doc,
      0
    },
    {0}  /* Sentinel */
};

#if PY_VERSION_HEX >= 0x03000000
#  define PYOBJECT_STR PyObject_Str
#else
#  define PYOBJECT_STR PyObject_Unicode
#endif

PyObject* PyBobLearnLinearMachine_Repr(PyBobLearnLinearMachineObject* self) {

  /**
   * Expected output:
   *
   * <xbob.learn.linear.Machine float64@(3, 2) [act: f(z) = tanh(z)]>
   */

  using bob::machine::IdentityActivation;

  static const std::string identity_str = IdentityActivation().str();

  auto weights = make_safe(PyBobLearnLinearMachine_getWeights(self, 0));
  if (!weights) return 0;
  auto dtype = make_safe(PyObject_GetAttrString(weights.get(), "dtype"));
  auto dtype_str = make_safe(PYOBJECT_STR(dtype.get()));
  auto shape = make_safe(PyObject_GetAttrString(weights.get(), "shape"));
  auto shape_str = make_safe(PyObject_Str(shape.get()));

  PyObject* retval = 0;

  if (self->cxx->getActivation()->str() == identity_str) {
    retval = PyUnicode_FromFormat("<%s %U@%U>",
        Py_TYPE(self)->tp_name, dtype_str.get(), shape_str.get());
  }

  else {
    retval = PyUnicode_FromFormat("<%s %s@%s [act: %s]>",
        Py_TYPE(self)->tp_name, dtype_str.get(), shape_str.get(),
        self->cxx->getActivation()->str().c_str());
  }

#if PYTHON_VERSION_HEX < 0x03000000
  if (!retval) return 0;
  PyObject* tmp = PyObject_Str(retval);
  Py_DECREF(retval);
  retval = tmp;
#endif

  return retval;

}

PyObject* PyBobLearnLinearMachine_Str(PyBobLearnLinearMachineObject* self) {

  /**
   * Expected output:
   *
   * xbob.learn.linear.Machine (float64) 3 inputs, 2 outputs [act: f(z) = C*z]
   *  subtract: [ 0.   0.5  0.5]
   *  divide: [ 0.5  1.   1. ]
   *  bias: [ 0.3 -3. ]
   *  [[ 0.4  0.1]
   *  [ 0.4  0.2]
   *  [ 0.2  0.7]]
   */

  using bob::machine::IdentityActivation;

  static const std::string identity_str = IdentityActivation().str();

  std::shared_ptr<PyObject> act;
  if (self->cxx->getActivation()->str() != identity_str) {
    act = make_safe(PyUnicode_FromFormat(" [act: %s]",
          self->cxx->getActivation()->str().c_str()));
  }
  else act = make_safe(PyUnicode_FromString(""));

  std::shared_ptr<PyObject> sub;
  if (blitz::any(self->cxx->getInputSubtraction())) {
    auto t = make_safe(PyBobLearnLinearMachine_getInputSubtraction(self, 0));
    auto t_str = make_safe(PYOBJECT_STR(t.get()));
    sub = make_safe(PyUnicode_FromFormat("\n subtract: %U", t_str.get()));
  }
  else sub = make_safe(PyUnicode_FromString(""));

  std::shared_ptr<PyObject> div;
  if (blitz::any(self->cxx->getInputDivision())) {
    auto t = make_safe(PyBobLearnLinearMachine_getInputDivision(self, 0));
    auto t_str = make_safe(PYOBJECT_STR(t.get()));
    div = make_safe(PyUnicode_FromFormat("\n divide: %U", t_str.get()));
  }
  else div = make_safe(PyUnicode_FromString(""));

  std::shared_ptr<PyObject> bias;
  if (blitz::any(self->cxx->getBiases())) {
    auto t = make_safe(PyBobLearnLinearMachine_getBiases(self, 0));
    auto t_str = make_safe(PYOBJECT_STR(t.get()));
    bias = make_safe(PyUnicode_FromFormat("\n bias: %U", t_str.get()));
  }
  else bias = make_safe(PyUnicode_FromString(""));

  auto weights = make_safe(PyBobLearnLinearMachine_getWeights(self, 0));
  if (!weights) return 0;
  auto weights_str = make_safe(PYOBJECT_STR(weights.get()));
  auto dtype = make_safe(PyObject_GetAttrString(weights.get(), "dtype"));
  auto dtype_str = make_safe(PYOBJECT_STR(dtype.get()));
  auto shape = make_safe(PyObject_GetAttrString(weights.get(), "shape"));

  PyObject* retval = PyUnicode_FromFormat("%s (%U) %" PY_FORMAT_SIZE_T "d inputs, %" PY_FORMAT_SIZE_T "d outputs%U%U%U%U\n %U",
    Py_TYPE(self)->tp_name, dtype_str.get(),
    PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape.get(), 0), PyExc_OverflowError),
    PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape.get(), 1), PyExc_OverflowError),
    act.get(), sub.get(), div.get(), bias.get(), weights_str.get());

#if PYTHON_VERSION_HEX < 0x03000000
  if (!retval) return 0;
  PyObject* tmp = PyObject_Str(retval);
  Py_DECREF(retval);
  retval = tmp;
#endif

  return retval;

}

PyDoc_STRVAR(s_forward_str, "forward");
PyDoc_STRVAR(s_forward_doc,
"o.forward(input [, output]) -> array\n\
\n\
Projects ``input`` through its internal weights and biases. If\n\
``output`` is provided, place output there instead of allocating\n\
a new array.\n\
\n\
The ``input`` (and ``output``) arrays can be either 1D or 2D\n\
64-bit float arrays. If one provides a 1D array, the ``output``\n\
array, if provided, should also be 1D, matching the output size\n\
of this machine. If one provides a 2D array, it is considered a\n\
set of vertically stacked 1D arrays (one input per row) and a\n\
2D array is produced or expected in ``output``. The ``output``\n\
array in this case shall have the same number of rows as the\n\
``input`` array and as many columns as the output size for this\n\
machine.\n\
\n\
.. note::\n\
\n\
   This method only accepts 64-bit float arrays as input or\n\
   output.\n\
\n");

static PyObject* PyBobLearnLinearMachine_forward
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {

  static const char* const_kwlist[] = {"input", "output", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  if (input->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output && output->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for output array `output'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (input->ndim < 1 || input->ndim > 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 1 or 2-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, input->ndim);
    return 0;
  }

  if (output && input->ndim != output->ndim) {
    PyErr_Format(PyExc_RuntimeError, "Input and output arrays should have matching number of dimensions, but input array `input' has %" PY_FORMAT_SIZE_T "d dimensions while output array `output' has %" PY_FORMAT_SIZE_T "d dimensions", input->ndim, output->ndim);
    return 0;
  }

  if (input->ndim == 1) {
    if (input->shape[0] != (Py_ssize_t)self->cxx->inputSize()) {
      PyErr_Format(PyExc_RuntimeError, "1D `input' array should have %" PY_FORMAT_SIZE_T "d elements matching `%s' input size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->inputSize(), Py_TYPE(self)->tp_name, input->shape[0]);
      return 0;
    }
    if (output && output->shape[0] != (Py_ssize_t)self->cxx->outputSize()) {
      PyErr_Format(PyExc_RuntimeError, "1D `output' array should have %" PY_FORMAT_SIZE_T "d elements matching `%s' output size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->outputSize(), Py_TYPE(self)->tp_name, output->shape[0]);
      return 0;
    }
  }
  else {
    if (input->shape[1] != (Py_ssize_t)self->cxx->inputSize()) {
      PyErr_Format(PyExc_RuntimeError, "2D `input' array should have %" PY_FORMAT_SIZE_T "d columns, matching `%s' input size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->inputSize(), Py_TYPE(self)->tp_name, input->shape[1]);
      return 0;
    }
    if (output && output->shape[1] != (Py_ssize_t)self->cxx->outputSize()) {
      PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d columns matching `%s' output size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->outputSize(), Py_TYPE(self)->tp_name, output->shape[1]);
      return 0;
    }
    if (output && input->shape[0] != output->shape[0]) {
      PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d rows matching `input' size, not %" PY_FORMAT_SIZE_T "d rows", input->shape[0], output->shape[0]);
      return 0;
    }
  }

  /** if ``output`` was not pre-allocated, do it now **/
  if (!output) {
    Py_ssize_t osize[2];
    if (input->ndim == 1) {
      osize[0] = self->cxx->outputSize();
    }
    else {
      osize[0] = input->shape[0];
      osize[1] = self->cxx->outputSize();
    }
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, input->ndim, osize);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the machine now **/
  try {
    if (input->ndim == 1) {
      self->cxx->forward_(*PyBlitzArrayCxx_AsBlitz<double,1>(input),
          *PyBlitzArrayCxx_AsBlitz<double,1>(output));
    }
    else {
      auto bzin = PyBlitzArrayCxx_AsBlitz<double,2>(input);
      auto bzout = PyBlitzArrayCxx_AsBlitz<double,2>(output);
      blitz::Range all = blitz::Range::all();
      for (int k=0; k<bzin->extent(0); ++k) {
        blitz::Array<double,1> i_ = (*bzin)(k, all);
        blitz::Array<double,1> o_ = (*bzout)(k, all);
        self->cxx->forward_(i_, o_); ///< no need to re-check
      }
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "%s cannot forward data: unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_INCREF(output);
  return PyBlitzArray_NUMPY_WRAP(reinterpret_cast<PyObject*>(output));

}

PyDoc_STRVAR(s_load_str, "load");
PyDoc_STRVAR(s_load_doc,
"o.load(f) -> None\n\
\n\
Loads itself from a :py:class:`xbob.io.HDF5File`\n\
\n\
");

static PyObject* PyBobLearnLinearMachine_Load
(PyBobLearnLinearMachineObject* self, PyObject* f) {

  if (!PyBobIoHDF5File_Check(f)) {
    PyErr_Format(PyExc_TypeError, "`%s' cannot load itself from `%s', only from an HDF5 file", Py_TYPE(self)->tp_name, Py_TYPE(f)->tp_name);
    return 0;
  }

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(f);

  try {
    self->cxx->load(*h5f->f);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot read data from file `%s' (at group `%s'): unknown exception caught", h5f->f->filename().c_str(),
        h5f->f->cwd().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_save_str, "save");
PyDoc_STRVAR(s_save_doc,
"o.save(f) -> None\n\
\n\
Saves itself at a :py:class:`xbob.io.HDF5File`\n\
\n\
");

static PyObject* PyBobLearnLinearMachine_Save
(PyBobLearnLinearMachineObject* self, PyObject* f) {

  if (!PyBobIoHDF5File_Check(f)) {
    PyErr_Format(PyExc_TypeError, "Activation function cannot write itself to `%s', only to an HDF5 file", Py_TYPE(f)->tp_name);
    return 0;
  }

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(f);

  try {
    self->cxx->save(*h5f->f);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot write data to file `%s' (at group `%s'): unknown exception caught", h5f->f->filename().c_str(),
        h5f->f->cwd().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_is_similar_to_str, "is_similar_to");
PyDoc_STRVAR(s_is_similar_to_doc,
"o.is_similar_to(other [, r_epsilon=1e-5 [, a_epsilon=1e-8]]) -> bool\n\
\n\
Compares this LinearMachine with the ``other`` one to be\n\
approximately the same.\n\
\n\
The optional values ``r_epsilon`` and ``a_epsilon`` refer to the\n\
relative and absolute precision for the ``weights``, ``biases``\n\
and any other values internal to this machine.\n\
\n\
");

static PyObject* PyBobLearnLinearMachine_IsSimilarTo
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", "r_epsilon", "a_epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnLinearMachine_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  auto other_ = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);

  if (self->cxx->is_similar_to(*other_->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_resize_str, "resize");
PyDoc_STRVAR(s_resize_doc,
"o.resize(input, output) -> None\n\
\n\
Resizes the machine. If either the input or output increases\n\
in size, the weights and other factors should be considered\n\
uninitialized. If the size is preserved or reduced, already\n\
initialized values will not be changed.\n\
\n\
.. note::\n\
\n\
   Use this method to force data compression. All will work\n\
   out given most relevant factors to be preserved are\n\
   organized on the top of the weight matrix. In this way,\n\
   reducing the system size will supress less relevant\n\
   projections.\n\
\n\
");

static PyObject* PyBobLearnLinearMachine_Resize
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"input", "output", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t input = 0;
  Py_ssize_t output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", kwlist,
        &input, &output)) return 0;

  try {
    self->cxx->resize(input, output);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot resize object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_RETURN_NONE;

}

static PyMethodDef PyBobLearnLinearMachine_methods[] = {
  {
    s_forward_str,
    (PyCFunction)PyBobLearnLinearMachine_forward,
    METH_VARARGS|METH_KEYWORDS,
    s_forward_doc
  },
  {
    s_load_str,
    (PyCFunction)PyBobLearnLinearMachine_Load,
    METH_O,
    s_load_doc
  },
  {
    s_save_str,
    (PyCFunction)PyBobLearnLinearMachine_Save,
    METH_O,
    s_save_doc
  },
  {
    s_is_similar_to_str,
    (PyCFunction)PyBobLearnLinearMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    s_is_similar_to_doc
  },
  {
    s_resize_str,
    (PyCFunction)PyBobLearnLinearMachine_Resize,
    METH_VARARGS|METH_KEYWORDS,
    s_resize_doc
  },
  {0} /* Sentinel */
};

static PyObject* PyBobLearnLinearMachine_new
(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobLearnLinearMachineObject* self =
    (PyBobLearnLinearMachineObject*)type->tp_alloc(type, 0);

  self->cxx = 0;

  return reinterpret_cast<PyObject*>(self);

}

PyObject* PyBobLearnLinearMachine_NewFromSize
(Py_ssize_t input, Py_ssize_t output) {

  PyBobLearnLinearMachineObject* retval = (PyBobLearnLinearMachineObject*)PyBobLearnLinearMachine_new(&PyBobLearnLinearMachine_Type, 0, 0);

  retval->cxx = new bob::machine::LinearMachine(input, output);

  return reinterpret_cast<PyObject*>(retval);

}

PyTypeObject PyBobLearnLinearMachine_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_linear_str,                                     /* tp_name */
    sizeof(PyBobLearnLinearMachineObject),            /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearMachine_delete,       /* tp_dealloc */
    0,                                                /* tp_print */
    0,                                                /* tp_getattr */
    0,                                                /* tp_setattr */
    0,                                                /* tp_compare */
    (reprfunc)PyBobLearnLinearMachine_Repr,           /* tp_repr */
    0,                                                /* tp_as_number */
    0,                                                /* tp_as_sequence */
    0,                                                /* tp_as_mapping */
    0,                                                /* tp_hash */
    (ternaryfunc)PyBobLearnLinearMachine_forward,     /* tp_call */
    (reprfunc)PyBobLearnLinearMachine_Str,            /* tp_str */
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
    PyBobLearnLinearMachine_new,                      /* tp_new */
};
