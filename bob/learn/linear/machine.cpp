/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 14 Jan 2014 14:26:09 CET
 *
 * @brief Bindings for a LinearMachine
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/defines.h>
#include <bob.io.base/api.h>
#include <bob.learn.activation/api.h>
#include <bob.learn.linear/api.h>
#include <bob.extension/documentation.h>
#include <structmember.h>

/**********************************************
 * Implementation of LinearMachine base class *
 **********************************************/

static auto Machine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".Machine",
  "A linear classifier, see C. M. Bishop, 'Pattern Recognition and Machine Learning', chapter 4 for more details.",
  "The basic matrix operation performed for projecting the input to the output is: :math:`o = w \\times i` (with :math:`w` being the vector of machine weights and :math:`i` the input data vector). "
  "The weights matrix is therefore organized column-wise. "
  "In this scheme, each column of the weights matrix can be interpreted as vector to which the input is projected. "
  "The number of columns of the weights matrix determines the number of outputs this linear machine will have. "
  "The number of rows is the number of allowed inputs it can process.\n\n"
  "Input and output is always performed on 1D arrays with 64-bit floating point numbers."
).add_constructor(bob::extension::FunctionDoc(
  "Machine",
  "Creates a new linear machine",
  "A linear machine can be constructed in different ways. "
  "In the first form, the user specifies optional input and output vector sizes. "
  "The machine is remains **uninitialized**. "
  "With the second form, the user passes a 2D array with 64-bit floats containing weight matrix to be used as the :py:attr:`weights` matrix by the new machine. "
  "In the third form the user passes a :py:class:`bob.io.base.HDF5File` opened for reading, which points to the machine information to be loaded in memory. "
  "Finally, in the last form (copy constructor), the user passes another :py:class:`bob.learn.linear.Machine` that will be deep copied."
)
.add_prototype("[input_size], [output_size])", "")
.add_prototype("weights", "")
.add_prototype("config", "")
.add_prototype("other", "")
.add_parameter("input_size", "int", "[Default: 0] The dimensionality of the input data that should be projected")
.add_parameter("output_size", "int", "[Default: 0] The dimensionality of the output data")
.add_parameter("weights", "array_like(2D, float)", "A weight matrix to initialize the :py:attr:`weights`")
.add_parameter("config", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file open for reading")
.add_parameter("other", ":py:class:`bob.learn.linear.Machine`", "The machine to copy construct")
);

static int PyBobLearnLinearMachine_init_sizes
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = Machine_doc.kwlist(0);

  Py_ssize_t input_size = 0;
  Py_ssize_t output_size = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|nn", kwlist,
        &input_size, &output_size)) return -1;

  self->cxx = new bob::learn::linear::Machine(input_size, output_size);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearMachine_init_weights(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = Machine_doc.kwlist(1);

  PyBlitzArrayObject* weights = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        &PyBlitzArray_Converter, &weights)) return -1;

  auto weights_ = make_safe(weights);
  if (weights->type_num != NPY_FLOAT64 || weights->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 2D arrays for property array `weights'", Py_TYPE(self)->tp_name);
    return -1;
  }

  self->cxx = new bob::learn::linear::Machine(*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearMachine_init_hdf5(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = Machine_doc.kwlist(2);

  PyObject* config = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobIoHDF5File_Type, &config)) return -1;

  auto h5f = reinterpret_cast<PyBobIoHDF5FileObject*>(config);
  self->cxx = new bob::learn::linear::Machine(*(h5f->f));
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearMachine_init_copy
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = Machine_doc.kwlist(3);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearMachine_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearMachineObject*>(other);
  self->cxx = new bob::learn::linear::Machine(*(copy->cxx));
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearMachine_init(PyBobLearnLinearMachineObject* self,
    PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 0: //default initializer
    case 2: //two sizes
      return PyBobLearnLinearMachine_init_sizes(self, args, kwds);

    case 1:{

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

        if (PyInt_Check(arg)) {
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

static auto weights = bob::extension::VariableDoc(
  "weights",
  "array_like(2D, float)",
  "Weight matrix to which the input is projected to",
  "The output of the projection is fed subject to bias and activation before being output"
);
static PyObject* PyBobLearnLinearMachine_getWeights
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getWeights()));
BOB_CATCH_MEMBER("weights", 0)
}

static int PyBobLearnLinearMachine_setWeights (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {
BOB_TRY
  PyBlitzArrayObject* weights = 0;
  if (!PyBlitzArray_Converter(o, &weights)) return -1;
  auto weights_ = make_safe(weights);

  if (weights->type_num != NPY_FLOAT64 || weights->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 2D arrays for property array `weights'", Py_TYPE(self)->tp_name);
    return -1;
  }

  self->cxx->setWeights(*PyBlitzArrayCxx_AsBlitz<double,2>(weights));
  return 0;
BOB_CATCH_MEMBER("weights", -1)
}

static auto biases = bob::extension::VariableDoc(
  "biases",
  "array_like(1D, float)",
  "Bias to the output units of this linear machine",
  "These values will be added to the output before the :py:attr:`activation` is applied. "
  "Must have the same size as :py:attr:`shape` [1]"
);
static PyObject* PyBobLearnLinearMachine_getBiases
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getBiases()));
BOB_CATCH_MEMBER("biases", 0)
}

static int PyBobLearnLinearMachine_setBiases (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {
BOB_TRY
  PyBlitzArrayObject* biases = 0;
  if (!PyBlitzArray_Converter(o, &biases)) return -1;
  auto biases_ = make_safe(biases);

  if (biases->type_num != NPY_FLOAT64 || biases->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `biases'", Py_TYPE(self)->tp_name);
    return -1;
  }

  self->cxx->setBiases(*PyBlitzArrayCxx_AsBlitz<double,1>(biases));
  return 0;
BOB_CATCH_MEMBER("biases", -1)
}

static auto input_subtract = bob::extension::VariableDoc(
  "input_subtract",
  "array_like(1D, float)",
  "Input subtraction factor",
  "These values will be subtracted before feeding data through the :py:attr:`weights` matrix. "
  "Must have the same size as :py:attr:`shape` [0]. "
  "By default, it is set to 0."
);
static PyObject* PyBobLearnLinearMachine_getInputSubtraction
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getInputSubtraction()));
BOB_CATCH_MEMBER("input_subtract", 0)
}

static int PyBobLearnLinearMachine_setInputSubtraction
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  PyBlitzArrayObject* input_subtract = 0;
  if (!PyBlitzArray_Converter(o, &input_subtract)) return -1;
  auto input_subtract_ = make_safe(input_subtract);

  if (input_subtract->type_num != NPY_FLOAT64 || input_subtract->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `input_subtract'", Py_TYPE(self)->tp_name);
    return -1;
  }

  self->cxx->setInputSubtraction(*PyBlitzArrayCxx_AsBlitz<double,1>(input_subtract));
  return 0;
BOB_CATCH_MEMBER("input_subtract", -1)
}


static auto input_divide = bob::extension::VariableDoc(
  "input_divide",
  "array_like(1D, float)",
  "Input division factor",
  "These data will be divided by ``input_divide`` before feeding it through the :py:attr:`weights` matrix. "
  "The division is applied just after subtraction. "
  "Must have the same size as :py:attr:`shape` [0]. "
  "By default, it is set to 1."
);
static PyObject* PyBobLearnLinearMachine_getInputDivision
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return PyBlitzArray_NUMPY_WRAP(PyBlitzArrayCxx_NewFromConstArray(self->cxx->getInputDivision()));
BOB_CATCH_MEMBER("input_divide", 0)
}

static int PyBobLearnLinearMachine_setInputDivision (PyBobLearnLinearMachineObject* self,
    PyObject* o, void* /*closure*/) {
BOB_TRY
  PyBlitzArrayObject* input_divide = 0;
  if (!PyBlitzArray_Converter(o, &input_divide)) return -1;
  auto input_divide_ = make_safe(input_divide);

  if (input_divide->type_num != NPY_FLOAT64 || input_divide->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit floats 1D arrays for property array `input_divide'", Py_TYPE(self)->tp_name);
    return -1;
  }

  self->cxx->setInputDivision(*PyBlitzArrayCxx_AsBlitz<double,1>(input_divide));
  return 0;
BOB_CATCH_MEMBER("input_divide", -1)
}


static auto shape = bob::extension::VariableDoc(
  "shape",
  "(int, int)",
  "The size of the :py:attr:`weights` matrix",
  "A tuple that represents the size of the input vector followed by the size of the output vector in the format ``(input, output)``."
);
static PyObject* PyBobLearnLinearMachine_getShape
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("(nn)", self->cxx->inputSize(), self->cxx->outputSize());
BOB_CATCH_MEMBER("shape", 0)
}

static int PyBobLearnLinearMachine_setShape
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
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

  self->cxx->resize(in, out);
  return 0;
BOB_CATCH_MEMBER("shape", -1)
}

static auto activation = bob::extension::VariableDoc(
  "activation",
  ":py:class:`bob.learn.activation.Activation` or one of its derivatives",
  "The activation function",
  "By default, the activation function is the :py:class:`bob.learn.activation.Identity` function."
);
static PyObject* PyBobLearnLinearMachine_getActivation
(PyBobLearnLinearMachineObject* self, void* /*closure*/) {
BOB_TRY
  return PyBobLearnActivation_NewFromActivation(self->cxx->getActivation());
BOB_CATCH_MEMBER("activation", 0)
}

static int PyBobLearnLinearMachine_setActivation
(PyBobLearnLinearMachineObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  if (!PyBobLearnActivation_Check(o)) {
    PyErr_Format(PyExc_TypeError, "%s activation requires an object of type `Activation' (or an inherited type), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  auto py = reinterpret_cast<PyBobLearnActivationObject*>(o);
  self->cxx->setActivation(py->cxx);
  return 0;
BOB_CATCH_MEMBER("activation", -1)
}

static PyGetSetDef PyBobLearnLinearMachine_getseters[] = {
    {
      weights.name(),
      (getter)PyBobLearnLinearMachine_getWeights,
      (setter)PyBobLearnLinearMachine_setWeights,
      weights.doc(),
      0
    },
    {
      biases.name(),
      (getter)PyBobLearnLinearMachine_getBiases,
      (setter)PyBobLearnLinearMachine_setBiases,
      biases.doc(),
      0
    },
    {
      input_subtract.name(),
      (getter)PyBobLearnLinearMachine_getInputSubtraction,
      (setter)PyBobLearnLinearMachine_setInputSubtraction,
      input_subtract.doc(),
      0
    },
    {
      input_divide.name(),
      (getter)PyBobLearnLinearMachine_getInputDivision,
      (setter)PyBobLearnLinearMachine_setInputDivision,
      input_divide.doc(),
      0
    },
    {
      shape.name(),
      (getter)PyBobLearnLinearMachine_getShape,
      (setter)PyBobLearnLinearMachine_setShape,
      shape.doc(),
      0
    },
    {
      activation.name(),
      (getter)PyBobLearnLinearMachine_getActivation,
      (setter)PyBobLearnLinearMachine_setActivation,
      activation.doc(),
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
   * <bob.learn.linear.Machine float64@(3, 2) [act: f(z) = tanh(z)]>
   */

  static const std::string identity_str = bob::learn::activation::IdentityActivation().str();

  auto weights = make_safe(PyBobLearnLinearMachine_getWeights(self, 0));
  if (!weights) return 0;
  auto dtype = make_safe(PyObject_GetAttrString(weights.get(), "dtype"));
  auto dtype_str = make_safe(PYOBJECT_STR(dtype.get()));
  auto shape = make_safe(PyObject_GetAttrString(weights.get(), "shape"));
  auto shape_str = make_safe(PYOBJECT_STR(shape.get()));

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
   * bob.learn.linear.Machine (float64) 3 inputs, 2 outputs [act: f(z) = C*z]
   *  subtract: [ 0.   0.5  0.5]
   *  divide: [ 0.5  1.   1. ]
   *  bias: [ 0.3 -3. ]
   *  [[ 0.4  0.1]
   *  [ 0.4  0.2]
   *  [ 0.2  0.7]]
   */

  static const std::string identity_str = bob::learn::activation::IdentityActivation().str();

  boost::shared_ptr<PyObject> act;
  if (self->cxx->getActivation()->str() != identity_str) {
    act = make_safe(PyUnicode_FromFormat(" [act: %s]",
          self->cxx->getActivation()->str().c_str()));
  }
  else act = make_safe(PyUnicode_FromString(""));

  boost::shared_ptr<PyObject> sub;
  if (blitz::any(self->cxx->getInputSubtraction())) {
    auto t = make_safe(PyBobLearnLinearMachine_getInputSubtraction(self, 0));
    auto t_str = make_safe(PYOBJECT_STR(t.get()));
    sub = make_safe(PyUnicode_FromFormat("\n subtract: %U", t_str.get()));
  }
  else sub = make_safe(PyUnicode_FromString(""));

  boost::shared_ptr<PyObject> div;
  if (blitz::any(self->cxx->getInputDivision())) {
    auto t = make_safe(PyBobLearnLinearMachine_getInputDivision(self, 0));
    auto t_str = make_safe(PYOBJECT_STR(t.get()));
    div = make_safe(PyUnicode_FromFormat("\n divide: %U", t_str.get()));
  }
  else div = make_safe(PyUnicode_FromString(""));

  boost::shared_ptr<PyObject> bias;
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

static auto forward = bob::extension::FunctionDoc(
  "forward",
  "Projects ``input`` through its internal weights and biases",
  "The ``input`` (and ``output``) arrays can be either 1D or 2D 64-bit float arrays. "
  "If one provides a 1D array, the ``output`` array, if provided, should also be 1D, matching the output size of this machine. "
  "If one provides a 2D array, it is considered a set of vertically stacked 1D arrays (one input per row) and a 2D array is produced or expected in ``output``. "
  "The ``output`` array in this case shall have the same number of rows as the ``input`` array and as many columns as the output size for this machine.\n\n"
  ".. note:: The ``__call__`` method is an alias for this method.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like(1D or 2D, float)", "The array that should be projected; must be compatible with :py:attr:`shape` [0]")
.add_parameter("output", "array_like(1D or 2D, float)", "The output array that will be filled. If given, must be compatible with ``input`` and :py:attr:`shape` [1]")
.add_return("output", "array_like(1D or 2D, float)", "The projected data; identical to the ``output`` parameter, if given")
;
static PyObject* PyBobLearnLinearMachine_forward
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = forward.kwlist();

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
  Py_INCREF(output);
  return PyBlitzArray_NUMPY_WRAP(reinterpret_cast<PyObject*>(output));
BOB_CATCH_MEMBER("forward", 0)
}

static auto load = bob::extension::FunctionDoc(
  "load",
  "Loads the machine from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file opened for reading")
;
static PyObject* PyBobLearnLinearMachine_Load(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = load.kwlist();
  PyBobIoHDF5FileObject* file;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &file)) return 0;

  auto file_ = make_safe(file);
  self->cxx->load(*file->f);
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("load", 0)
}


static auto save = bob::extension::FunctionDoc(
  "save",
  "Saves the machine to the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;
static PyObject* PyBobLearnLinearMachine_Save(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = save.kwlist();
  PyBobIoHDF5FileObject* file;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,"O&", kwlist, PyBobIoHDF5File_Converter, &file)) return 0;

  auto file_ = make_safe(file);
  self->cxx->save(*file->f);
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("save", 0)
}


static auto is_similar_to = bob::extension::FunctionDoc(
  "is_similar_to",
  "Compares this LinearMachine with the ``other`` one to be approximately the same",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the relative and absolute precision for the :py:attr:`weights`, :py:attr:`biases` and any other values internal to this machine.",
  true
)
.add_prototype("other, [r_epsilon], [a_epsilon]", "similar")
.add_parameter("other", ":py:class:`bob.learn.linear.Machine`", "The other machine to compare with")
.add_parameter("r_epsilon", "float", "[Default: ``1e-5``] The relative precision")
.add_parameter("a_epsilon", "float", "[Default: ``1e-8``] The absolute precision")
.add_return("similar", "bool", "``True`` if the ``other`` machine is similar to this one, otherwise ``False``")
;
static PyObject* PyBobLearnLinearMachine_IsSimilarTo
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = is_similar_to.kwlist();

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
BOB_CATCH_MEMBER("is_similar_to", 0)
}

static auto resize = bob::extension::FunctionDoc(
  "resize",
  "Resizes the machine",
  "If either the input or output increases in size, the weights and other factors should be considered uninitialized. "
  "If the size is preserved or reduced, already initialized values will not be changed.\n\n"
  ".. note::\n\n"
  "   Use this method to force data compression.\n"
  "   All will work out given most relevant factors to be preserved are organized on the top of the weight matrix.\n"
  "   In this way, reducing the system size will suppress less relevant projections.",
  true
)
.add_prototype("input, output")
.add_parameter("input", "int", "The input dimension to be set")
.add_parameter("output", "int", "The output dimension to be set")
;
static PyObject* PyBobLearnLinearMachine_Resize
(PyBobLearnLinearMachineObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = resize.kwlist();

  Py_ssize_t input, output;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", kwlist, &input, &output)) return 0;

  self->cxx->resize(input, output);
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("resize", 0)
}

static PyMethodDef PyBobLearnLinearMachine_methods[] = {
  {
    forward.name(),
    (PyCFunction)PyBobLearnLinearMachine_forward,
    METH_VARARGS|METH_KEYWORDS,
    forward.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobLearnLinearMachine_Load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobLearnLinearMachine_Save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {
    is_similar_to.name(),
    (PyCFunction)PyBobLearnLinearMachine_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    is_similar_to.doc()
  },
  {
    resize.name(),
    (PyCFunction)PyBobLearnLinearMachine_Resize,
    METH_VARARGS|METH_KEYWORDS,
    resize.doc()
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

  retval->cxx = new bob::learn::linear::Machine(input, output);

  return reinterpret_cast<PyObject*>(retval);

}

// Linear Machine
PyTypeObject PyBobLearnLinearMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearMachine(PyObject* module)
{
  // Linear Machine
  PyBobLearnLinearMachine_Type.tp_name = Machine_doc.name();
  PyBobLearnLinearMachine_Type.tp_basicsize = sizeof(PyBobLearnLinearMachineObject);
  PyBobLearnLinearMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearMachine_Type.tp_doc = Machine_doc.doc();

  // set the functions
  PyBobLearnLinearMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearMachine_init);
  PyBobLearnLinearMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearMachine_delete);
  PyBobLearnLinearMachine_Type.tp_methods = PyBobLearnLinearMachine_methods;
  PyBobLearnLinearMachine_Type.tp_getset = PyBobLearnLinearMachine_getseters;
  PyBobLearnLinearMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnLinearMachine_forward);
  PyBobLearnLinearMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearMachine_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearMachine_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearMachine_Type);
  return PyModule_AddObject(module, "Machine", (PyObject*)&PyBobLearnLinearMachine_Type) >= 0;
}
