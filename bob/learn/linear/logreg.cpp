/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 14:27:40 CET
 *
 * @brief Python bindings to CGLogReg trainer
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.learn.linear/api.h>
#include <bob.extension/documentation.h>
#include <structmember.h>

/************************************************
 * Implementation of CGLogRegTrainer base class *
 ************************************************/

static auto CGLogReg_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".CGLogRegTrainer",
  "Trains a linear machine to perform Linear Logistic Regression",
  "The training stage will place the resulting weights (and bias) in a linear machine with a single output dimension. "
  "For details about Linear Logistic Regression, please see:\n\n"
  "1. A comparison of numerical optimizers for logistic regression, T. Minka, (`See Microsoft Research paper <http://research.microsoft.com/en-us/um/people/minka/papers/logreg/>`_)\n"
  "2. FoCal, https://sites.google.com/site/nikobrummer/focal"
).add_constructor(bob::extension::FunctionDoc(
  "CGLogRegTrainer",
  "Creates a new trainer to perform Linear Logistic Regression",
  "There are two initializers for objects of this class. "
  "In the first variant, the user passes the discrete training parameters, including the classes prior, convergence threshold and the maximum number of conjugate gradient (CG) iterations among other parameters. "
  "If ``mean_std_norm`` is set to ``True``, your input data will be mean/standard-deviation normalized and the according values will be set as normalization factors to the resulting machine. "
  "The second initialization form copy constructs a new trainer from an existing one."
)
.add_prototype("[prior], [convergence_threshold], [max_iterations], [reg], [mean_std_norm]", "")
.add_prototype("other", "")
.add_parameter("prior", "float", "[Default: ``0.5``] The synthetic prior (should be in range :math:`]0.,1.[`)")
.add_parameter("convergence_threshold", "float", "[Default: ``1e-5``] The convergence threshold for the conjugate gradient algorithm")
.add_parameter("max_iterations", "int", "[Default: ``10000``] The maximum number of iterations for the conjugate gradient algorithm")
.add_parameter("reg", "float", "[Default: ``0.``] The regularization factor lambda. If you set this to the value of ``0.``, then the algorithm will apply **no** regularization whatsoever")\
.add_parameter("mean_std_norm", "bool", "[Default: ``False``] Performs mean and standard-deviation normalization (whitening) of the input data before training the (resulting) :py:class:`bob.learn.linear.Machine`. Setting this to ``True`` is recommended for large data sets with significant amplitude variations between dimensions")
.add_parameter("other", ":py:class:`CGLogRegTrainer`", "If you decide to copy construct from another object of the same type, pass it using this parameter")
);
static int PyBobLearnLinearCGLogRegTrainer_init_parameters
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = CGLogReg_doc.kwlist(0);

  double prior = 0.5;
  double convergence_threshold = 1e-5;
  Py_ssize_t max_iterations = 10000;
  double lambda = 0.;
  PyObject* mean_std_norm = Py_False;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ddndO", kwlist,
        &prior, &convergence_threshold, &max_iterations,
        &lambda, &mean_std_norm)) return -1;

  int mean_std_norm_ = PyObject_IsTrue(mean_std_norm);
  if (mean_std_norm_ == -1) return -1; //error on conversion

  self->cxx = new bob::learn::linear::CGLogRegTrainer(prior, convergence_threshold, max_iterations, lambda, mean_std_norm_);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearCGLogRegTrainer_init_copy
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = CGLogReg_doc.kwlist(1);

  PyBobLearnLinearCGLogRegTrainerObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearCGLogRegTrainer_Type, &other)) return -1;

  self->cxx = new bob::learn::linear::CGLogRegTrainer(*other->cxx);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBobLearnLinearCGLogRegTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearCGLogRegTrainer_Type));
}

static int PyBobLearnLinearCGLogRegTrainer_init
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {

  PyObject* arg = 0; ///< borrowed (don't delete)
  if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
  else {
    if (!kwds)
      return PyBobLearnLinearCGLogRegTrainer_init_parameters(self, args, kwds);
    PyObject* tmp = PyDict_Values(kwds);
    auto tmp_ = make_safe(tmp);
    arg = PyList_GET_ITEM(tmp, 0);
  }

  if (PyBobLearnLinearCGLogRegTrainer_Check(arg)) {
    return PyBobLearnLinearCGLogRegTrainer_init_copy(self, args, kwds);
  }

  return PyBobLearnLinearCGLogRegTrainer_init_parameters(self, args, kwds);
}

static void PyBobLearnLinearCGLogRegTrainer_delete
(PyBobLearnLinearCGLogRegTrainerObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyBobLearnLinearCGLogRegTrainer_RichCompare
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearCGLogRegTrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearCGLogRegTrainerObject*>(other);

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
  "Trains a linear machine to perform linear logistic regression",
  "The resulting machine will have the same number of inputs as columns in ``negatives`` and ``positives`` and a single output. "
  "This method always returns a machine, which will be identical to the one provided (if the user passed one) or a new one allocated internally.",
  true
)
.add_prototype("negatives, positives, [machine]", "machine")
.add_parameter("negatives, positives", "array_like(2D, float)", "``negatives`` and ``positives`` should be arrays organized in such a way that every row corresponds to a new observation of the phenomena (i.e., a new sample) and every column corresponds to a different feature")
.add_parameter("machine", ":py:class:`bob.learn.linear.Machine`", "The user may provide or not a machine that will be set by this method. If provided, the machine should have 1 output and the  number of inputs matching the number of columns in the input data arrays")
.add_return("machine", ":py:class:`bob.learn.linear.Machine`", "The trained linear machine; identical to the ``machine`` parameter, if given")
;
static PyObject* PyBobLearnLinearCGLogRegTrainer_Train
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = train.kwlist();

  PyBlitzArrayObject* negatives = 0;
  PyBlitzArrayObject* positives = 0;
  PyBobLearnLinearMachineObject* machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&|O!", kwlist,
        &PyBlitzArray_Converter, &negatives,
        &PyBlitzArray_Converter, &positives,
        &PyBobLearnLinearMachine_Type, &machine
        ))
    return 0;

  auto negatives_ = make_safe(negatives); ///< auto-delete in case of problems
  auto positives_ = make_safe(positives); ///< auto-delete in case of problems

  if (negatives->ndim != 2 || negatives->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input array `negatives'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (positives->ndim != 2 || positives->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input array `positives'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (negatives->shape[1] != positives->shape[1]) {
    PyErr_Format(PyExc_TypeError, "`%s' requires input matrices `negatives' and `positives' to have the same number of columns (i.e. feature dimensions) but `negatives' has %" PY_FORMAT_SIZE_T "d columns and `positives' has %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, negatives->shape[1], positives->shape[1]);
    return 0;
  }

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyBobLearnLinearMachineObject> machine_;
  if (!machine) {
    machine = reinterpret_cast<PyBobLearnLinearMachineObject*>(PyBobLearnLinearMachine_NewFromSize(negatives->shape[1], 1));
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  self->cxx->train(*machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(negatives), *PyBlitzArrayCxx_AsBlitz<double,2>(positives));

  return Py_BuildValue("O", machine);
BOB_CATCH_MEMBER("train", 0)
}

static PyMethodDef PyBobLearnLinearCGLogRegTrainer_methods[] = {
  {
    train.name(),
    (PyCFunction)PyBobLearnLinearCGLogRegTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    train.doc()
  },
  {0} /* Sentinel */
};


static auto prior = bob::extension::VariableDoc(
  "prior",
  "float",
  "The synthetic prior (should be in range :math:`]0.,1.[`)"
);
static PyObject* PyBobLearnLinearCGLogRegTrainer_getPrior
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("d", self->cxx->getPrior());
BOB_CATCH_MEMBER("prior", 0)
}

static int PyBobLearnLinearCGLogRegTrainer_setPrior
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;
  self->cxx->setPrior(v);
  return 0;
BOB_CATCH_MEMBER("prior", -1)
}


static auto convergence_threshold = bob::extension::VariableDoc(
  "convergence_threshold",
  "float",
  "The convergence threshold for the conjugate gradient algorithm"
);
static PyObject* PyBobLearnLinearCGLogRegTrainer_getConvergenceThreshold
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("d", self->cxx->getConvergenceThreshold());
BOB_CATCH_MEMBER("convergence_threshold", 0)
}

static int PyBobLearnLinearCGLogRegTrainer_setConvergenceThreshold
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  self->cxx->setConvergenceThreshold(v);
  return 0;
BOB_CATCH_MEMBER("convergence_threshold", -1)
}


static auto max_iterations = bob::extension::VariableDoc(
  "max_iterations",
  "int",
  "The maximum number of iterations for the conjugate gradient algorithm"
);
static PyObject* PyBobLearnLinearCGLogRegTrainer_getMaxIterations
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getMaxIterations());
BOB_CATCH_MEMBER("max_iterations", 0)
}

static int PyBobLearnLinearCGLogRegTrainer_setMaxIterations
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (v < 0) return -1;

  self->cxx->setMaxIterations(v);
  return 0;
BOB_CATCH_MEMBER("max_iterations", -1)
}


static auto reg = bob::extension::VariableDoc(
  "reg",
  "float",
  "The regularization factor lambda",
  "If you set this to the value of ``0.``, the algorithm will apply **no** regularization whatsoever."
);
static PyObject* PyBobLearnLinearCGLogRegTrainer_getLambda
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("d", self->cxx->getLambda());
BOB_CATCH_MEMBER("reg", 0)
}

static int PyBobLearnLinearCGLogRegTrainer_setLambda
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  self->cxx->setLambda(v);
  return 0;
BOB_CATCH_MEMBER("reg", -1)
}


static auto whiten = bob::extension::VariableDoc(
  "mean_std_norm",
  "bool",
  "Perform whitening on input data?",
  "If set to ``True``, performs mean and standard-deviation normalization (whitening) of the input data before training the (resulting) Machine. "
  "Setting this to ``True`` is recommended for large data sets with significant amplitude variations between dimensions"
);
static PyObject* PyBobLearnLinearCGLogRegTrainer_getNorm
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {
BOB_TRY
  if (self->cxx->getNorm()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER("mean_std_norm", 0)
}

static int PyBobLearnLinearCGLogRegTrainer_setNorm
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  int istrue = PyObject_IsTrue(o);
  if (istrue == -1) return -1;

  self->cxx->setNorm(istrue);
  return 0;
BOB_CATCH_MEMBER("mean_std_norm", -1)
}

static PyGetSetDef PyBobLearnLinearCGLogRegTrainer_getseters[] = {
    {
      prior.name(),
      (getter)PyBobLearnLinearCGLogRegTrainer_getPrior,
      (setter)PyBobLearnLinearCGLogRegTrainer_setPrior,
      prior.doc(),
      0
    },
    {
      convergence_threshold.name(),
      (getter)PyBobLearnLinearCGLogRegTrainer_getConvergenceThreshold,
      (setter)PyBobLearnLinearCGLogRegTrainer_setConvergenceThreshold,
      convergence_threshold.doc(),
      0
    },
    {
      max_iterations.name(),
      (getter)PyBobLearnLinearCGLogRegTrainer_getMaxIterations,
      (setter)PyBobLearnLinearCGLogRegTrainer_setMaxIterations,
      max_iterations.doc(),
      0
    },
    {
      reg.name(),
      (getter)PyBobLearnLinearCGLogRegTrainer_getLambda,
      (setter)PyBobLearnLinearCGLogRegTrainer_setLambda,
      reg.doc(),
      0
    },
    {
      whiten.name(),
      (getter)PyBobLearnLinearCGLogRegTrainer_getNorm,
      (setter)PyBobLearnLinearCGLogRegTrainer_setNorm,
      whiten.doc(),
      0
    },
    {0}  /* Sentinel */
};

// Linear Logistic Regression Trainer
PyTypeObject PyBobLearnLinearCGLogRegTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearCGLogReg(PyObject* module)
{
  // Linear Logistic Regression Trainer
  PyBobLearnLinearCGLogRegTrainer_Type.tp_name = CGLogReg_doc.name();
  PyBobLearnLinearCGLogRegTrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearCGLogRegTrainerObject);
  PyBobLearnLinearCGLogRegTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearCGLogRegTrainer_Type.tp_doc = CGLogReg_doc.doc();

  // set the functions
  PyBobLearnLinearCGLogRegTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearCGLogRegTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearCGLogRegTrainer_init);
  PyBobLearnLinearCGLogRegTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearCGLogRegTrainer_delete);
  PyBobLearnLinearCGLogRegTrainer_Type.tp_methods = PyBobLearnLinearCGLogRegTrainer_methods;
  PyBobLearnLinearCGLogRegTrainer_Type.tp_getset = PyBobLearnLinearCGLogRegTrainer_getseters;
  PyBobLearnLinearCGLogRegTrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearCGLogRegTrainer_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearCGLogRegTrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearCGLogRegTrainer_Type);
  return PyModule_AddObject(module, "CGLogRegTrainer", (PyObject*)&PyBobLearnLinearCGLogRegTrainer_Type) >= 0;
}
