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
#include <structmember.h>

/************************************************
 * Implementation of CGLogRegTrainer base class *
 ************************************************/

PyDoc_STRVAR(s_logregtrainer_str, BOB_EXT_MODULE_PREFIX ".CGLogRegTrainer");

PyDoc_STRVAR(s_logregtrainer_doc,
"CGLogRegTrainer([prior=0.5, [convergence_threshold=1e-5,\n\
                 [max_iterations=10000, [reg=0.,\n\
                 [mean_std_norm=False]]]]]) -> new CGLogRegTrainer\n\
\n\
CGLogRegTrainer(other) -> new CGLogRegTrainer\n\
\n\
Trains a linear machine to perform Linear Logistic Regression.\n\
\n\
There are two initializers for objects of this class. In the\n\
first variant, the user passes the discrete training parameters,\n\
including the classes prior, convergence threshold and the\n\
maximum number of conjugate gradient (CG) iterations among\n\
other parameters. The second initialization form copy constructs\n\
a new trainer from an existing one.\n\
\n\
The training stage will place the resulting weights (and bias)\n\
in a linear machine with a single output dimension. If the\n\
parameter ``mean_std_norm`` is set to ``True``, then your input\n\
data will be mean/standard-deviation normalized and the\n\
according values will be set as normalization factors to the\n\
resulting machine.\n\
\n\
Keyword parameters:\n\
\n\
prior, float (optional)\n\
  The synthetic prior (should be in range :math:`]0.,1.[`).\n\
\n\
convergence_threshold, float (optional)\n\
  The convergence threshold for the conjugate gradient algorithm\n\
\n\
max_iterations, int (optional)\n\
  The maximum number of iterations for the conjugate gradient\n\
  algorithm\n\
\n\
reg, float (optional)\n\
  The regularization factor lambda. If you set this to the value of\n\
  ``0.0`` (the default), then the algorithm will apply **no**\n\
  regularization whatsoever.\n\
\n\
mean_std_norm, bool (optional)\n\
  Performs mean and standard-deviation normalization (whitening)\n\
  of the input data before training the (resulting) Machine.\n\
  Setting this to ``True`` is recommended for large data sets with\n\
  significant amplitude variations between dimensions.\n\
\n\
other, CGLogRegTrainer\n\
  If you decide to copy construct from another object of the same\n\
  type, pass it using this parameter.\n\
\n\
References:\n\
\n\
1. A comparison of numerical optimizers for logistic regression,\n\
   T. Minka, (`See Microsoft Research paper\n\
   <http://research.microsoft.com/en-us/um/people/minka/papers/logreg/>`_)\n\
2. FoCal, https://sites.google.com/site/nikobrummer/focal\n\
\n\
");

static int PyBobLearnLinearCGLogRegTrainer_init_parameters
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "prior",
    "convergence_threshold",
    "max_iterations",
    "reg",
    "mean_std_norm",
    0
  };
  static char** kwlist = const_cast<char**>(const_kwlist);

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

  try {
    self->cxx = new bob::learn::linear::CGLogRegTrainer(prior,
        convergence_threshold, max_iterations, lambda, mean_std_norm_?true:false);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot initialize object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static int PyBobLearnLinearCGLogRegTrainer_init_copy
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearCGLogRegTrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearCGLogRegTrainerObject*>(other);

  try {
    self->cxx = new bob::learn::linear::CGLogRegTrainer(*(copy->cxx));
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

PyDoc_STRVAR(s_train_str, "train");
PyDoc_STRVAR(s_train_doc,
"o.train(negatives, positives [, machine]) -> machine\n\
\n\
Trains a linear machine to perform linear logistic regression.\n\
\n\
The resulting machine will have the same number of inputs as\n\
columns in ``negatives`` and ``positives`` and a single output.\n\
\n\
Keyword parameters:\n\
\n\
negatives, positives, 2D 64-bit float arrays\n\
  These should be arrays organized in such a way that every row\n\
  corresponds to a new observation of the phenomena (i.e., a new\n\
  sample) and every column corresponds to a different feature.\n\
\n\
machine, :py:class:`Machine` (optional)\n\
  The user may provide or not an object of type\n\
  :py:class:`bob.learn.linear.Machine` that will be set by this\n\
  method. If provided, the machine should have 1 output and the\n\
  correct number of inputs matching the number of columns in the\n\
  input data arrays.\n\
\n\
This method always returns a machine, which will be the same as\n\
the one provided (if the user passed one) or a new one allocated\n\
internally.\n\
\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_Train
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"negatives", "positives", "machine", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* negatives = 0;
  PyBlitzArrayObject* positives = 0;
  PyObject* machine = 0;

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
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(negatives->shape[1], 1);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  try {
    self->cxx->train(
        *pymac->cxx,
        *PyBlitzArrayCxx_AsBlitz<double,2>(negatives),
        *PyBlitzArrayCxx_AsBlitz<double,2>(positives)
        );
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

static PyMethodDef PyBobLearnLinearCGLogRegTrainer_methods[] = {
  {
    s_train_str,
    (PyCFunction)PyBobLearnLinearCGLogRegTrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    s_train_doc
  },
  {0} /* Sentinel */
};

PyDoc_STRVAR(s_prior_str, "prior");
PyDoc_STRVAR(s_prior_doc,
"The synthetic prior (should be in range :math:`]0.,1.[`).\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_getPrior
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {

  return Py_BuildValue("d", self->cxx->getPrior());

}

static int PyBobLearnLinearCGLogRegTrainer_setPrior
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {

  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->cxx->setPrior(v);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot set prior on `%s' with value `%g': unknown exception caught", Py_TYPE(self)->tp_name, v);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_convergence_threshold_str, "convergence_threshold");
PyDoc_STRVAR(s_convergence_threshold_doc,
"The convergence threshold for the conjugate gradient algorithm\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_getConvergenceThreshold
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {

  return Py_BuildValue("d", self->cxx->getConvergenceThreshold());

}

static int PyBobLearnLinearCGLogRegTrainer_setConvergenceThreshold
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {

  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->cxx->setConvergenceThreshold(v);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot set convergence threshold on `%s' with value `%g': unknown exception caught", Py_TYPE(self)->tp_name, v);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_max_iterations_str, "max_iterations");
PyDoc_STRVAR(s_max_iterations_doc,
"The maximum number of iterations for the conjugate gradient algorithm\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_getMaxIterations
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {

  return Py_BuildValue("n", self->cxx->getMaxIterations());

}

static int PyBobLearnLinearCGLogRegTrainer_setMaxIterations
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {

  Py_ssize_t v = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (v < 0) return -1;

  try {
    self->cxx->setMaxIterations(v);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot set max_iterations on `%s' with value `%" PY_FORMAT_SIZE_T "d': unknown exception caught", Py_TYPE(self)->tp_name, v);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_lambda_str, "reg");
PyDoc_STRVAR(s_lambda_doc,
"The regularization factor lambda. If you set this to the value of\n\
``0.0`` (the default), then the algorithm will apply **no**\n\
regularization whatsoever.\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_getLambda
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {

  return Py_BuildValue("d", self->cxx->getLambda());

}

static int PyBobLearnLinearCGLogRegTrainer_setLambda
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {

  double v = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->cxx->setLambda(v);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot set reg on `%s' with value `%g': unknown exception caught", Py_TYPE(self)->tp_name, v);
    return -1;
  }

  return 0;

}

PyDoc_STRVAR(s_norm_str, "mean_std_norm");
PyDoc_STRVAR(s_norm_doc,
"Performs mean and standard-deviation normalization (whitening)\n\
of the input data before training the (resulting) Machine.\n\
Setting this to ``True`` is recommended for large data sets with\n\
significant amplitude variations between dimensions.\n\
");

static PyObject* PyBobLearnLinearCGLogRegTrainer_getNorm
(PyBobLearnLinearCGLogRegTrainerObject* self, void* /*closure*/) {

  if (self->cxx->getNorm()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static int PyBobLearnLinearCGLogRegTrainer_setNorm
(PyBobLearnLinearCGLogRegTrainerObject* self, PyObject* o, void* /*closure*/) {

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  if (istrue) self->cxx->setNorm(true);
  else self->cxx->setNorm(false);

  return 0;

}

static PyGetSetDef PyBobLearnLinearCGLogRegTrainer_getseters[] = {
    {
      s_prior_str,
      (getter)PyBobLearnLinearCGLogRegTrainer_getPrior,
      (setter)PyBobLearnLinearCGLogRegTrainer_setPrior,
      s_prior_doc,
      0
    },
    {
      s_convergence_threshold_str,
      (getter)PyBobLearnLinearCGLogRegTrainer_getConvergenceThreshold,
      (setter)PyBobLearnLinearCGLogRegTrainer_setConvergenceThreshold,
      s_convergence_threshold_doc,
      0
    },
    {
      s_max_iterations_str,
      (getter)PyBobLearnLinearCGLogRegTrainer_getMaxIterations,
      (setter)PyBobLearnLinearCGLogRegTrainer_setMaxIterations,
      s_max_iterations_doc,
      0
    },
    {
      s_lambda_str,
      (getter)PyBobLearnLinearCGLogRegTrainer_getLambda,
      (setter)PyBobLearnLinearCGLogRegTrainer_setLambda,
      s_lambda_doc,
      0
    },
    {
      s_norm_str,
      (getter)PyBobLearnLinearCGLogRegTrainer_getNorm,
      (setter)PyBobLearnLinearCGLogRegTrainer_setNorm,
      s_norm_doc,
      0
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearCGLogRegTrainer_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_logregtrainer_str,                                     /* tp_name */
    sizeof(PyBobLearnLinearCGLogRegTrainerObject),           /* tp_basicsize */
    0,                                                       /* tp_itemsize */
    (destructor)PyBobLearnLinearCGLogRegTrainer_delete,      /* tp_dealloc */
    0,                                                       /* tp_print */
    0,                                                       /* tp_getattr */
    0,                                                       /* tp_setattr */
    0,                                                       /* tp_compare */
    0,                                                       /* tp_repr */
    0,                                                       /* tp_as_number */
    0,                                                       /* tp_as_sequence */
    0,                                                       /* tp_as_mapping */
    0,                                                       /* tp_hash */
    0,                                                       /* tp_call */
    0,                                                       /* tp_str */
    0,                                                       /* tp_getattro */
    0,                                                       /* tp_setattro */
    0,                                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                /* tp_flags */
    s_logregtrainer_doc,                                     /* tp_doc */
    0,                                                       /* tp_traverse */
    0,                                                       /* tp_clear */
    (richcmpfunc)PyBobLearnLinearCGLogRegTrainer_RichCompare, /* tp_richcompare */
    0,                                                       /* tp_weaklistoffset */
    0,                                                       /* tp_iter */
    0,                                                       /* tp_iternext */
    PyBobLearnLinearCGLogRegTrainer_methods,                 /* tp_methods */
    0,                                                       /* tp_members */
    PyBobLearnLinearCGLogRegTrainer_getseters,               /* tp_getset */
    0,                                                       /* tp_base */
    0,                                                       /* tp_dict */
    0,                                                       /* tp_descr_get */
    0,                                                       /* tp_descr_set */
    0,                                                       /* tp_dictoffset */
    (initproc)PyBobLearnLinearCGLogRegTrainer_init,          /* tp_init */
    0,                                                       /* tp_alloc */
    0,                                                       /* tp_new */
};
