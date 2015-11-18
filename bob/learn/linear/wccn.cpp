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
#include <bob.extension/documentation.h>
#include <structmember.h>

/*************************************************
 * Implementation of WCCNTrainer base class *
 *************************************************/

static auto WCCN_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".WCCNTrainer",
  "Trains a linear machine to perform Within-Class Covariance Normalization (WCCN)",
  "WCCN finds the projection matrix W that allows us to linearly project the data matrix X to another (sub) space such that:\n\n"
  ".. math::\n\n"
  "   (1/N) S_{w} = W W^T\n\n"
  "where :math:`W` is an upper triangular matrix computed using Cholesky Decomposition:\n\n"
  ".. math::\n\n"
  "   W = cholesky([(1/K) S_{w} ]^{-1})\n\n"
  "where:\n\n"
  ":math:`K`\n\n  the number of classes\n\n"
  ":math:`S_w`\n\n the within-class scatter; it also has dimensions ``(X.shape[0], X.shape[0])`` and is defined as :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`, with :math:`C_k` being a set representing all samples for class k.\n\n"
  ":math:`m_k`\n\n  the class *k* empirical mean, defined as :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`\n\n"
  "References:\n\n"
  "1. Within-class covariance normalization for SVM-based speaker recognition, Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke, In INTERSPEECH, 2006.\n"
  "2. http://en.wikipedia.org/wiki/Cholesky_decomposition"
)
.add_constructor(bob::extension::FunctionDoc(
  "WCCNTrainer",
  "Constructs a new trainer to train a linear machine to perform WCCN"
)
.add_prototype("","")
.add_prototype("other","")
.add_parameter("other", ":py:class:`WCCNTrainer`", "Another WCCN trainer to copy")
);
static int PyBobLearnLinearWCCNTrainer_init_default
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  self->cxx = new bob::learn::linear::WCCNTrainer();
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearWCCNTrainer_init_copy
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = WCCN_doc.kwlist(1);

  PyBobLearnLinearWCCNTrainerObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearWCCNTrainer_Type, &other)) return -1;

  self->cxx = new bob::learn::linear::WCCNTrainer(*other->cxx);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBobLearnLinearWCCNTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearWCCNTrainer_Type));
}

static int PyBobLearnLinearWCCNTrainer_init(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  if (nargs == 1) {
    return PyBobLearnLinearWCCNTrainer_init_copy(self, args, kwds);
  }

  return PyBobLearnLinearWCCNTrainer_init_default(self, args, kwds);
}

static void PyBobLearnLinearWCCNTrainer_delete(PyBobLearnLinearWCCNTrainerObject* self) {
  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnLinearWCCNTrainer_RichCompare(PyBobLearnLinearWCCNTrainerObject* self, PyObject* other, int op) {
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


static auto train = bob::extension::FunctionDoc(
  "train",
  "Trains a linear machine using WCCN",
  "The value of ``X`` should be a sequence over as many 2D 64-bit floating point number arrays as classes in the problem. "
  "All arrays will be checked for conformance (identical number of columns). "
  "To accomplish this, either prepare a list with all your class observations organized in 2D arrays or pass a 3D array in which the first dimension (depth) contains as many elements as classes you want to train for.\n\n"
  "The resulting machine will have the same number of inputs **and** outputs as columns in any of ``X``'s matrices.\n\n"
  "The user may provide or not an object of type :py:class:`bob.learn.linear.Machine` that will be set by this method. "
  "In such a case, the machine should have a shape that matches ``(X.shape[1], X.shape[1])``. "
  "If the user does not provide a machine to be set, then a new one will be allocated internally. "
  "In both cases, the resulting machine is always returned.",
  true
)
.add_prototype("X, [machine]", "machine")
.add_parameter("X", "[array_like(2D,float)] or array_like(3D, float)", "The training data arranged by class")
.add_parameter("machine", ":py:class:`bob.learn.linear.Machine`", "A pre-allocated machine to be trained; may be omitted")
.add_return("machine", ":py:class:`bob.learn.linear.Machine`", "The trained machine; identical to the ``machine`` parameter, if specified")
;
static PyObject* PyBobLearnLinearWCCNTrainer_Train
(PyBobLearnLinearWCCNTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = train.kwlist();

  PyObject* X = 0;
  PyBobLearnLinearMachineObject* machine = 0;

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
  boost::shared_ptr<PyBobLearnLinearMachineObject> machine_;
  if (!machine) {
    machine = reinterpret_cast<PyBobLearnLinearMachineObject*>(PyBobLearnLinearMachine_NewFromSize(ncol, ncol));
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  self->cxx->train(*machine->cxx, Xseq);

  return Py_BuildValue("O", machine);
BOB_CATCH_MEMBER("train", 0)
}

static PyMethodDef PyBobLearnLinearWCCNTrainer_methods[] = {
  {
    train.name(),
    (PyCFunction)PyBobLearnLinearWCCNTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    train.doc()
  },
  {0} /* Sentinel */
};


// WCCN Trainer
PyTypeObject PyBobLearnLinearWCCNTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearWCCN(PyObject* module)
{
  // WCCN Trainer
  PyBobLearnLinearWCCNTrainer_Type.tp_name = WCCN_doc.name();
  PyBobLearnLinearWCCNTrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearWCCNTrainerObject);
  PyBobLearnLinearWCCNTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearWCCNTrainer_Type.tp_doc = WCCN_doc.doc();

  // set the functions
  PyBobLearnLinearWCCNTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearWCCNTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearWCCNTrainer_init);
  PyBobLearnLinearWCCNTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearWCCNTrainer_delete);
  PyBobLearnLinearWCCNTrainer_Type.tp_methods = PyBobLearnLinearWCCNTrainer_methods;
  PyBobLearnLinearWCCNTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearWCCNTrainer_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearWCCNTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearWCCNTrainer_Type);
  return PyModule_AddObject(module, "WCCNTrainer", (PyObject*)&PyBobLearnLinearWCCNTrainer_Type) >= 0;
}
