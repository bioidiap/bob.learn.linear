/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 14:27:40 CET
 *
 * @brief Python bindings to Whitening trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob/config.h>
#include <bob.learn.linear/api.h>
#include <structmember.h>

/*************************************************
 * Implementation of WhiteningTrainer base class *
 *************************************************/

PyDoc_STRVAR(s_whiteningtrainer_str, BOB_EXT_MODULE_PREFIX ".WhiteningTrainer");

PyDoc_STRVAR(s_whiteningtrainer_doc,
"WhiteningTrainer() -> new WhiteningTrainer\n\
\n\
Trains a linear machine` to perform Cholesky Whitening.\n\
\n\
The whitening transformation is a decorrelation method that converts\n\
the covariance matrix of a set of samples into the identity matrix\n\
:math:`I`. This effectively linearly transforms random variables such\n\
that the resulting variables are uncorrelated and have the same\n\
variances as the original random variables. This transformation is\n\
invertible. The method is called the whitening transform because it\n\
transforms the input matrix X closer towards white noise (let's call\n\
it :math:`\\tilde{X}`): \n\
\n\
.. math::\n\
   \n\
   Cov(\\tilde{X}) = I\n\
\n\
where:\n\
\n\
.. math::\n\
   \n\
   \\tilde{X} = X W\n\
\n\
W is the projection matrix that allows us to linearly project the data\n\
matrix X to another (sub) space such that:\n\
\n\
.. math::\n\
   \n\
   Cov(X) = W W^T\n\
\n\
W is computed using Cholesky Decomposition:\n\
\n\
.. math::\n\
   \n\
   W = cholesky([Cov(X)]^{-1})\n\
\n\
References:\n\
\n\
1. https://rtmath.net/help/html/e9c12dc0-e813-4ca9-aaa3-82340f1c5d24.htm\n\
2. http://en.wikipedia.org/wiki/Cholesky_decomposition\n\
\n\
");

static int PyBobLearnLinearWhiteningTrainer_init_default
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {

  try {
    self->cxx = new bob::learn::linear::WhiteningTrainer();
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

static int PyBobLearnLinearWhiteningTrainer_init_copy
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearWhiteningTrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearWhiteningTrainerObject*>(other);

  try {
    self->cxx = new bob::learn::linear::WhiteningTrainer(*(copy->cxx));
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

int PyBobLearnLinearWhiteningTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearWhiteningTrainer_Type));
}

static int PyBobLearnLinearWhiteningTrainer_init
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  if (nargs == 1) {
    return PyBobLearnLinearWhiteningTrainer_init_copy(self, args, kwds);
  }

  return PyBobLearnLinearWhiteningTrainer_init_default(self, args, kwds);

}

static void PyBobLearnLinearWhiteningTrainer_delete
(PyBobLearnLinearWhiteningTrainerObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject* PyBobLearnLinearWhiteningTrainer_RichCompare
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearWhiteningTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearWhiteningTrainerObject*>(other);

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

PyDoc_STRVAR(s_is_similar_to_str, "is_similar_to");
PyDoc_STRVAR(s_is_similar_to_doc,
"o.is_similar_to(other [, r_epsilon=1e-5 [, a_epsilon=1e-8]]) -> bool\n\
\n\
Compares this WhiteningTrainer with the ``other`` one to be\n\
approximately the same.\n\
\n\
The optional values ``r_epsilon`` and ``a_epsilon`` refer to the\n\
relative and absolute precision for the ``weights``, ``biases``\n\
and any other values internal to this machine.\n\
\n\
");

static PyObject* PyBobLearnLinearWhiteningTrainer_IsSimilarTo
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", "r_epsilon", "a_epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnLinearWhiteningTrainer_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  auto other_ = reinterpret_cast<PyBobLearnLinearWhiteningTrainerObject*>(other);

  if (self->cxx->is_similar_to(*other_->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_train_str, "train");
PyDoc_STRVAR(s_train_doc,
"o.train(X [, machine]) -> machine\n\
\n\
The resulting machine will have the same number of inputs\n\
**and** outputs as columns in ``X``.\n\
\n\
The user may provide or not an object of type\n\
:py:class:`bob.learn.linear.Machine` that will be set by this\n\
method. In such a case, the machine should have a shape that\n\
matches ``(X.shape[1], X.shape[1])``. If the user does not\n\
provide a machine to be set, then a new one will be allocated\n\
internally. In both cases, the resulting machine is always\n\
returned by this method.\n\
\n\
The input data matrix :math:`X` should correspond to a 64-bit\n\
floating point 2D array organized in such a way that every row\n\
corresponds to a new observation of the phenomena (i.e., a new\n\
sample) and every column corresponds to a different feature.\n\
\n\
");

static PyObject* PyBobLearnLinearWhiteningTrainer_Train
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"X", "machine", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* X = 0;
  PyObject* machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O!", kwlist,
        &PyBlitzArray_Converter, &X,
        &PyBobLearnLinearMachine_Type, &machine
        ))
    return 0;

  auto X_ = make_safe(X); ///< auto-delete in case of problems

  if (X->ndim != 2 || X->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input array `X'", Py_TYPE(self)->tp_name);
    return 0;
  }

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(X->shape[1], X->shape[1]);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  try {
    self->cxx->train(*pymac->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(X));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot train `%s' with this `%s': unknown exception caught", Py_TYPE(machine)->tp_name, Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_INCREF(machine);
  return machine;

}

static PyMethodDef PyBobLearnLinearWhiteningTrainer_methods[] = {
  {
    s_is_similar_to_str,
    (PyCFunction)PyBobLearnLinearWhiteningTrainer_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    s_is_similar_to_doc
  },
  {
    s_train_str,
    (PyCFunction)PyBobLearnLinearWhiteningTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    s_train_doc
  },
  {0} /* Sentinel */
};

static PyGetSetDef PyBobLearnLinearWhiteningTrainer_getseters[] = {
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearWhiteningTrainer_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_whiteningtrainer_str,                           /* tp_name */
    sizeof(PyBobLearnLinearWhiteningTrainerObject),   /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearWhiteningTrainer_delete, /* tp_dealloc */
    0,                                                /* tp_print */
    0,                                                /* tp_getattr */
    0,                                                /* tp_setattr */
    0,                                                /* tp_compare */
    0,                                                /* tp_repr */
    0,                                                /* tp_as_number */
    0,                                                /* tp_as_sequence */
    0,                                                /* tp_as_mapping */
    0,                                                /* tp_hash */
    0,                                                /* tp_call */
    0,                                                /* tp_str */
    0,                                                /* tp_getattro */
    0,                                                /* tp_setattro */
    0,                                                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,         /* tp_flags */
    s_whiteningtrainer_doc,                           /* tp_doc */
    0,                                                /* tp_traverse */
    0,                                                /* tp_clear */
    (richcmpfunc)PyBobLearnLinearWhiteningTrainer_RichCompare, /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    PyBobLearnLinearWhiteningTrainer_methods,         /* tp_methods */
    0,                                                /* tp_members */
    PyBobLearnLinearWhiteningTrainer_getseters,       /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)PyBobLearnLinearWhiteningTrainer_init,  /* tp_init */
    0,                                                /* tp_alloc */
    0,                                                /* tp_new */
};
