/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 14:27:40 CET
 *
 * @brief Python bindings to WCCN trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/config.h>
#include <bob.learn.linear/api.h>
#include <structmember.h>

/*************************************************
 * Implementation of WCCNTrainer base class *
 *************************************************/

PyDoc_STRVAR(s_wccntrainer_str, BOB_EXT_MODULE_PREFIX ".WCCNTrainer");

PyDoc_STRVAR(s_wccntrainer_doc,
"WCCNTrainer() -> new WCCNTrainer\n\
\n\
Trains a linear machine to perform Within-Class Covariance Normalisation (WCCN).\n\
\n\
WCCN finds the projection matrix W that allows us to linearly\n\
project the data matrix X to another (sub) space such that:\n\
\n\
.. math::\n\
   \n\
   (1/N) S_{w} = W W^T\n\
\n\
where W is an upper triangular matrix computed using Cholesky\n\
Decomposition:\n\
\n\
.. math::\n\
   \n\
   W = cholesky([(1/K) S_{w} ]^{-1})\n\
\n\
where:\n\
\n\
:math:`K`\n\
\n\
  the number of classes\n\
\n\
:math:`S_w`\n\
\n\
   the within-class scatter; it also has dimensions\n\
   ``(X.shape[0], X.shape[0])`` and is defined as\n\
   :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`,\n\
   :math:`C_k` a set representing all samples for class k.\n\
\n\
:math:`m_k`\n\
  \n\
   the class *k* empirical mean, defined as\n\
   :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`\n\
\n\
References:\n\
\n\
1. Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke, Within-class covariance normalization for SVM-based speaker recognition, In INTERSPEECH, 2006.\n\
2. http://en.wikipedia.org/wiki/Cholesky_decomposition\n\
\n\
");

static int PyBobLearnLinearWCCNTrainer_init_default
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {

  try {
    self->cxx = new bob::learn::linear::WCCNTrainer();
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

static int PyBobLearnLinearWCCNTrainer_init_copy
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearWCCNTrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearWCCNTrainerObject*>(other);

  try {
    self->cxx = new bob::learn::linear::WCCNTrainer(*(copy->cxx));
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

int PyBobLearnLinearWCCNTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearWCCNTrainer_Type));
}

static int PyBobLearnLinearWCCNTrainer_init
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  if (nargs == 1) {
    return PyBobLearnLinearWCCNTrainer_init_copy(self, args, kwds);
  }

  return PyBobLearnLinearWCCNTrainer_init_default(self, args, kwds);

}

static void PyBobLearnLinearWCCNTrainer_delete
(PyBobLearnLinearWCCNTrainerObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject* PyBobLearnLinearWCCNTrainer_RichCompare
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearWCCNTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearWCCNTrainerObject*>(other);

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
Compares this WCCNTrainer with the ``other`` one to be\n\
approximately the same.\n\
\n\
The optional values ``r_epsilon`` and ``a_epsilon`` refer to the\n\
relative and absolute precision for the ``weights``, ``biases``\n\
and any other values internal to this machine.\n\
\n\
");

static PyObject* PyBobLearnLinearWCCNTrainer_IsSimilarTo
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", "r_epsilon", "a_epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnLinearWCCNTrainer_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  auto other_ = reinterpret_cast<PyBobLearnLinearWCCNTrainerObject*>(other);

  if (self->cxx->is_similar_to(*other_->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_train_str, "train");
PyDoc_STRVAR(s_train_doc,
"o.train(X [, machine]) -> machine\n\
\n\
Trains a linear machine using WCCN.\n\
\n\
The resulting machine will have the same number of inputs\n\
**and** outputs as columns in any of ``X``'s matrices.\n\
\n\
The user may provide or not an object of type\n\
:py:class:`bob.learn.linear.Machine` that will be set by this\n\
method. In such a case, the machine should have a shape that\n\
matches ``(X.shape[1], X.shape[1])``. If the user does not\n\
provide a machine to be set, then a new one will be allocated\n\
internally. In both cases, the resulting machine is always\n\
returned by this method.\n\
\n\
The value of ``X`` should be a sequence over as many 2D 64-bit\n\
floating point number arrays as classes in the problem. All\n\
arrays will be checked for conformance (identical number of\n\
columns). To accomplish this, either prepare a list with all\n\
your class observations organised in 2D arrays or pass a 3D\n\
array in which the first dimension (depth) contains as many\n\
elements as classes you want to train for.\n\
\n\
");

static PyObject* PyBobLearnLinearWCCNTrainer_Train
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"X", "machine", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* X = 0;
  PyObject* machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O!", kwlist,
        &X, &PyBobLearnLinearMachine_Type, &machine)) return 0;

  /**
  // Note: strangely, if you pass dict.values(), this check does not work
  if (!PyIter_Check(X)) {
    PyErr_Format(PyExc_TypeError, "`%s' requires an iterable for parameter `X', but you passed `%s' which does not implement the iterator protocol", Py_TYPE(self)->tp_name, Py_TYPE(X)->tp_name);
    return 0;
  }
  **/

  /* Checks and converts all entries */
  std::vector<blitz::Array<double,2> > Xseq;
  std::vector<boost::shared_ptr<PyBlitzArrayObject>> Xseq_;

  PyObject* iterator = PyObject_GetIter(X);
  if (!iterator) return 0;
  auto iterator_ = make_safe(iterator);

  while (PyObject* item = PyIter_Next(iterator)) {
    auto item_ = make_safe(item);

    PyBlitzArrayObject* bz = 0;

    if (!PyBlitzArray_Converter(item, &bz)) {
      PyErr_Format(PyExc_TypeError, "`%s' could not convert object of type `%s' at position %" PY_FORMAT_SIZE_T "d of input sequence `X' into an array - check your input", Py_TYPE(self)->tp_name, Py_TYPE(item)->tp_name, Xseq.size());
      return 0;
    }

    if (bz->ndim != 2 || bz->type_num != NPY_FLOAT64) {
      PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input sequence `X' (or any other object coercible to that), but at position %" PY_FORMAT_SIZE_T "d I have found an object with %" PY_FORMAT_SIZE_T "d dimensions and with type `%s' which is not compatible - check your input", Py_TYPE(self)->tp_name, Xseq.size(), bz->ndim, PyBlitzArray_TypenumAsString(bz->type_num));
      Py_DECREF(bz);
      return 0;
    }

    Xseq_.push_back(make_safe(bz)); ///< prevents data deletion
    Xseq.push_back(*PyBlitzArrayCxx_AsBlitz<double,2>(bz)); ///< only a view!
  }

  if (PyErr_Occurred()) return 0;

  if (Xseq.size() < 2) {
    PyErr_Format(PyExc_RuntimeError, "`%s' requires an iterable for parameter `X' leading to, at least, two entries (representing two classes), but you have passed something that has only %" PY_FORMAT_SIZE_T "d entries", Py_TYPE(self)->tp_name, Xseq.size());
    return 0;
  }

  // checks all elements in X have the same number of columns
  Py_ssize_t ncol = Xseq_[0]->shape[1];
  for (Py_ssize_t k=1; k<(Py_ssize_t)Xseq.size(); ++k) {
    if (Xseq_[k]->shape[1] != ncol) {
      PyErr_Format(PyExc_RuntimeError, "`%s' requires all matrices in input sequence `X' to have the same number of columns, but entry at position 0 has %" PY_FORMAT_SIZE_T "d columns white entry in position %" PY_FORMAT_SIZE_T "d has %" PY_FORMAT_SIZE_T "d columns", Py_TYPE(self)->tp_name, ncol, k, Xseq_[k]->shape[1]);
      return 0;
    }
  }

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(ncol, ncol);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  try {
    self->cxx->train(*pymac->cxx, Xseq);
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

static PyMethodDef PyBobLearnLinearWCCNTrainer_methods[] = {
  {
    s_is_similar_to_str,
    (PyCFunction)PyBobLearnLinearWCCNTrainer_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    s_is_similar_to_doc
  },
  {
    s_train_str,
    (PyCFunction)PyBobLearnLinearWCCNTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    s_train_doc
  },
  {0} /* Sentinel */
};

static PyGetSetDef PyBobLearnLinearWCCNTrainer_getseters[] = {
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearWCCNTrainer_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_wccntrainer_str,                                /* tp_name */
    sizeof(PyBobLearnLinearWCCNTrainerObject),        /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearWCCNTrainer_delete,   /* tp_dealloc */
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
    s_wccntrainer_doc,                                /* tp_doc */
    0,                                                /* tp_traverse */
    0,                                                /* tp_clear */
    (richcmpfunc)PyBobLearnLinearWCCNTrainer_RichCompare, /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    PyBobLearnLinearWCCNTrainer_methods,              /* tp_methods */
    0,                                                /* tp_members */
    PyBobLearnLinearWCCNTrainer_getseters,            /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)PyBobLearnLinearWCCNTrainer_init,       /* tp_init */
    0,                                                /* tp_alloc */
    0,                                                /* tp_new */
};
