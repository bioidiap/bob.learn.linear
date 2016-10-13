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
#include <bob.core/config.h>
#include <bob.learn.linear/api.h>
#include <bob.extension/documentation.h>
#include <structmember.h>

/*************************************************
 * Implementation of WhiteningTrainer base class *
 *************************************************/

static auto Whitening_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".WhiteningTrainer",
  "Trains a linear :py:class:`bob.learn.linear.Machine` to perform Cholesky whitening.",
  "The whitening transformation is a decorrelation method that converts the covariance matrix of a set of samples into the identity matrix :math:`I`. "
  "This effectively linearly transforms random variables such that the resulting variables are uncorrelated and have the same variances as the original random variables. "
  "This transformation is invertible. "
  "The method is called the whitening transform because it transforms the input matrix :math:`X` closer towards white noise (let's call it :math:`\\tilde{X}`):\n\n"
  ".. math::\n\n   Cov(\\tilde{X}) = I\n\n"
  "with:\n\n"
  ".. math::\n\n   \\tilde{X} = X W\n\n"
  "where :math:`W` is the projection matrix that allows us to linearly project the data matrix :math:`X` to another (sub) space such that:\n\n"
  ".. math::\n\n   Cov(X) = W W^T\n\n"
  ":math:`W` is computed using Cholesky decomposition:\n\n"
  ".. math::\n\n   W = cholesky([Cov(X)]^{-1})\n\n"
  "References:\n\n"
  "1. https://rtmath.net/help/html/e9c12dc0-e813-4ca9-aaa3-82340f1c5d24.htm\n"
  "2. http://en.wikipedia.org/wiki/Cholesky_decomposition"
).add_constructor(bob::extension::FunctionDoc(
  "WhiteningTrainer",
  "Constructs a new whitening trainer"
)
.add_prototype("","")
.add_prototype("other","")
.add_parameter("other", ":py:class:`WhiteningTrainer`", "Another whitening trainer to copy")
);

static int PyBobLearnLinearWhiteningTrainer_init_default
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  self->cxx = new bob::learn::linear::WhiteningTrainer();
  return 0;
BOB_CATCH_MEMBER("constructor",-1)
}

static int PyBobLearnLinearWhiteningTrainer_init_copy
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = Whitening_doc.kwlist(1);

  PyBobLearnLinearWhiteningTrainerObject* other;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearWhiteningTrainer_Type, &other)) return -1;

  self->cxx = new bob::learn::linear::WhiteningTrainer(*(other->cxx));
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
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


static auto train = bob::extension::FunctionDoc(
  "train",
  "Trains a linear machine to perform Cholesky whitening",
  "The user may provide or not an object of type :py:class:`bob.learn.linear.Machine` that will be set by this method. "
  "In such a case, the machine should have a shape that matches ``(X.shape[1], X.shape[1])``. "
  "If the user does not provide a machine to be set, then a new one will be allocated internally. "
  "In both cases, the resulting machine is always returned by this method.\n\n"
  "The input data matrix :math:`X` should correspond to a 64-bit floating point 2D array organized in such a way that every row corresponds to a new observation of the phenomena (i.e., a new sample) and every column corresponds to a different feature.",
  true
)
.add_prototype("X, [machine]", "machine")
.add_parameter("X", "array_like(2D, float)", "The training data")
.add_parameter("machine", ":py:class:`bob.learn.linear.Machine`", "A pre-allocated machine to be trained; may be omitted")
.add_return("machine", ":py:class:`bob.learn.linear.Machine`", "The trained machine; identical to the ``machine`` parameter, if specified")
;
static PyObject* PyBobLearnLinearWhiteningTrainer_Train
(PyBobLearnLinearWhiteningTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = train.kwlist();

  PyBlitzArrayObject* X;
  PyBobLearnLinearMachineObject* machine = 0;

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
  boost::shared_ptr<PyBobLearnLinearMachineObject> machine_;
  if (!machine) {
    machine = reinterpret_cast<PyBobLearnLinearMachineObject*>(PyBobLearnLinearMachine_NewFromSize(X->shape[1], X->shape[1]));
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  // perform training
  self->cxx->train(*machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(X));
  return Py_BuildValue("O", machine);
BOB_CATCH_MEMBER("train", 0)
}

static PyMethodDef PyBobLearnLinearWhiteningTrainer_methods[] = {
  {
    train.name(),
    (PyCFunction)PyBobLearnLinearWhiteningTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    train.doc()
  },
  {0} /* Sentinel */
};


// Whitening Trainer
PyTypeObject PyBobLearnLinearWhiteningTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearWhitening(PyObject* module)
{
  // Whitening Trainer
  PyBobLearnLinearWhiteningTrainer_Type.tp_name = Whitening_doc.name();
  PyBobLearnLinearWhiteningTrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearWhiteningTrainerObject);
  PyBobLearnLinearWhiteningTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearWhiteningTrainer_Type.tp_doc = Whitening_doc.doc();

  // set the functions
  PyBobLearnLinearWhiteningTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearWhiteningTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearWhiteningTrainer_init);
  PyBobLearnLinearWhiteningTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearWhiteningTrainer_delete);
  PyBobLearnLinearWhiteningTrainer_Type.tp_methods = PyBobLearnLinearWhiteningTrainer_methods;
  PyBobLearnLinearWhiteningTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearWhiteningTrainer_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearWhiteningTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearWhiteningTrainer_Type);
  return PyModule_AddObject(module, "WhiteningTrainer", (PyObject*)&PyBobLearnLinearWhiteningTrainer_Type) >= 0;
}
