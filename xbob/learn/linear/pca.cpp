/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 14:27:40 CET
 *
 * @brief Python bindings to PCA trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define XBOB_LEARN_LINEAR_MODULE
#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>
#include <bob/config.h>
#include <bob/trainer/PCATrainer.h>
#include <xbob.learn.linear/api.h>
#include <structmember.h>

/*******************************************
 * Implementation of PCATrainer base class *
 *******************************************/

PyDoc_STRVAR(s_pcatrainer_str, XBOB_EXT_MODULE_PREFIX ".PCATrainer");

PyDoc_STRVAR(s_pcatrainer_doc,
"PCATrainer([use_svd=True]) -> new PCATrainer\n\
PCATrainer(other) -> new PCATrainer\n\
\n\
Sets a linear machine to perform the Principal Component\n\
Analysis (a.k.a. Karhunen-Loeve Transform) on a given dataset\n\
using either Singular Value Decomposition (SVD, *the\n\
default*) or the Covariance Matrix Method.\n\
\n\
The training stage will place the resulting principal\n\
components in the linear machine and set it up to extract the\n\
variable means automatically. As an option, you may preset\n\
the trainer so that the normalization performed by the\n\
resulting linear machine also divides the variables by the\n\
standard deviation of each variable ensemble.\n\
\n\
There are two initializers for objects of this class. In the\n\
first variant, the user can pass flag indicating if the trainer\n\
should use SVD (default) or the covariance method for PCA\n\
extraction. The second initialization form copy constructs a\n\
new trainer from an existing one.\n\
\n\
The principal components correspond the direction of the data\n\
in which its points are maximally spread.\n\
\n\
Computing these principal components is equivalent to\n\
computing the eigen vectors U for the covariance matrix\n\
Sigma extracted from the data matrix X. The covariance matrix\n\
for the data is computed using the equation below:\n\
\n\
.. math::\n\
   \n\
   \\Sigma &= \\frac{((X-\\mu_X)^T(X-\\mu_X))}{m-1} \\text{ with}\\\\\n\
   \\mu_X  &= \\sum_i^N x_i\n\
\n\
where :math:`m` is the number of rows in :math:`X` (that is,\n\
the number of samples).\n\
\n\
Once you are in possession of :math:`\\Sigma`, it suffices\n\
to compute the eigen vectors U, solving the linear equation:\n\
\n\
.. math::\n\
   \n\
   (\\Sigma - e I) U = 0\n\
\n\
In this trainer, we make use of LAPACK's ``dsyevd`` to solve\n\
the above equation, if you choose to use the Covariance\n\
Method for extracting the principal components of your data\n\
matrix :math:`X`.\n\
\n\
*By default* though, this class will perform PC extraction\n\
using SVD. SVD is a factorization technique that allows for\n\
the decomposition of a matrix :math:`X`, with size (m,n) into\n\
3 other matrices in this way:\n\
\n\
.. math::\n\
   \n\
   X = U S V^*\n\
\n\
where:\n\
\n\
:math:`U`\n\
  unitary matrix of size (m,m) - a.k.a., left singular\n\
  vectors of X\n\
\n\
:math:`S`\n\
  rectangular diagonal matrix with nonnegative real\n\
  numbers, size (m,n)\n\
\n\
:math:`V^*`\n\
  (the conjugate transpose of V) unitary matrix of size\n\
  (n,n), a.k.a. right singular vectors of X\n\
\n\
We can use this property to avoid the computation of the\n\
covariance matrix of the data matrix :math:`X`, if we note\n\
the following:\n\
\n\
.. math::\n\
   \n\
   X &= U S V^* \\text{ , so} \\\\\n\
   XX^T &= U S V^* V S U^*\\\\\n\
   XX^T &= U S^2 U^*\n\
\n\
If X has zero mean, we can conclude by inspection that the\n\
U matrix obtained by SVD contains the eigen vectors of the\n\
covariance matrix of X (:math:`XX^T`) and :math:`S^2/(m-1)`\n\
corresponds to its eigen values.\n\
\n\
.. note::\n\
   \n\
   Our implementation uses LAPACK's ``dgesdd`` to compute the\n\
   solution to this linear equation.\n\
\n\
The corresponding :py:class:`xbob.learn.Linear.Machine` and\n\
returned eigen-values of :math:`\\Sigma`, are pre-sorted in\n\
descending order (the first eigen-vector - or column - of the\n\
weight matrix in the :py:class:`xbob.learn.Linear.Machine`\n\
corresponds to the highest eigen value obtained).\n\
\n\
.. note::\n\
   \n\
   One question you should pose yourself is which of the\n\
   methods to choose. Here is some advice: you should prefer\n\
   the covariance method over SVD when the number of samples\n\
   (rows of :math:`X`) is greater than the number of features\n\
   (columns of :math:`X`). It provides a faster execution\n\
   path in that case. Otherwise, use the *default* SVD method.\n\
\n\
References:\n\
\n\
1. Eigenfaces for Recognition, Turk & Pentland, Journal of\n\
   Cognitive Neuroscience (1991) Volume: 3, Issue: 1,\n\
   Publisher: MIT Press, Pages: 71-86\n\
2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n\
3. http://en.wikipedia.org/wiki/Principal_component_analysis\n\
4. http://www.netlib.org/lapack/double/dsyevd.f\n\
5. http://www.netlib.org/lapack/double/dgesdd.f\n\
"
);

static int PyBobLearnLinearPCATrainer_init_bool
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"use_svd", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* use_svd = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &use_svd)) return -1;

  int use_svd_ = PyObject_IsTrue(use_svd);

  if (use_svd_ == -1) return -1;

  try {
    self->cxx = new bob::trainer::PCATrainer(use_svd_?true:false);
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

static int PyBobLearnLinearPCATrainer_init_copy
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearPCATrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearPCATrainerObject*>(other);

  try {
    self->cxx = new bob::trainer::PCATrainer(*(copy->cxx));
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

int PyBobLearnLinearPCATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearPCATrainer_Type));
}

static int PyBobLearnLinearPCATrainer_init
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 0: //default initializer
      return PyBobLearnLinearPCATrainer_init_bool(self, args, kwds);

    case 1:

      {

        PyObject* arg = 0; ///< borrowed (don't delete)
        if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
        else {
          PyObject* tmp = PyDict_Values(kwds);
          auto tmp_ = make_safe(tmp);
          arg = PyList_GET_ITEM(tmp, 0);
        }

        if (PyBobLearnLinearPCATrainer_Check(arg)) {
          return PyBobLearnLinearPCATrainer_init_copy(self, args, kwds);
        }
        else {
          return PyBobLearnLinearPCATrainer_init_bool(self, args, kwds);
        }

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);

      }

      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - `%s' requires 0 or 1 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);

  }

  return -1;

}

static void PyBobLearnLinearPCATrainer_delete
(PyBobLearnLinearPCATrainerObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject* PyBobLearnLinearPCATrainer_RichCompare
(PyBobLearnLinearPCATrainerObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearPCATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearPCATrainerObject*>(other);

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
Compares this PCATrainer with the ``other`` one to be\n\
approximately the same.\n\
\n\
The optional values ``r_epsilon`` and ``a_epsilon`` refer to the\n\
relative and absolute precision for the ``weights``, ``biases``\n\
and any other values internal to this machine.\n\
\n\
");

static PyObject* PyBobLearnLinearPCATrainer_IsSimilarTo
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", "r_epsilon", "a_epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnLinearPCATrainer_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  auto other_ = reinterpret_cast<PyBobLearnLinearPCATrainerObject*>(other);

  if (self->cxx->is_similar_to(*other_->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_train_str, "train");
PyDoc_STRVAR(s_train_doc,
"o.train(X [, machine]) -> (machine, eigen_values)\n\
\n\
Trains a linear machine to perform the KLT.\n\
\n\
The resulting machine will have the same number of inputs as\n\
columns in ``X`` and :math:`K` eigen-vectors, where\n\
:math:`K=\\min{(S-1,F)}`, with :math:`S` being the number of\n\
rows in ``X`` (samples) and :math:`F` the number of columns\n\
(or features). The vectors are arranged by decreasing\n\
eigen-value automatically. You don't need to sort the results.\n\
\n\
The user may provide or not an object of type\n\
:py:class:`xbob.learn.linear.Machine` that will be set by this\n\
method. If provided, machine should have the correct number of\n\
inputs and outputs matching, respectively, the number of columns\n\
in the input data array ``X`` and the output of the method\n\
:py:meth:`xbob.learn.linear.PCATrainer.output_size` (see\n\
help).\n\
\n\
The input data matrix :math:`X` should correspond to a 64-bit\n\
floating point array organized in such a way that every row\n\
corresponds to a new observation of the phenomena (i.e., a new\n\
sample) and every column corresponds to a different feature.\n\
\n\
This method returns a tuple consisting of the trained machine\n\
and a 1D 64-bit floating point array containing the eigen-values\n\
calculated while computing the KLT. The eigen-value ordering\n\
matches that of eigen-vectors set in the machine.\n\
\n\
");

static PyObject* PyBobLearnLinearPCATrainer_Train
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

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

  // evaluates the expected rank for the output, allocate eigens value array
  auto X_bz = PyBlitzArrayCxx_AsBlitz<double,2>(X);
  Py_ssize_t rank = self->cxx->output_size(*X_bz);
  auto eigval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &rank);
  auto eigval_ = make_safe(eigval); ///< auto-delete in case of problems

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(X_bz->extent(1), rank);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  try {
    auto eigval_bz = PyBlitzArrayCxx_AsBlitz<double,1>(reinterpret_cast<PyBlitzArrayObject*>(eigval));
    self->cxx->train(*pymac->cxx, *eigval_bz, *X_bz);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot train `%s' with this `%s': unknown exception caught", Py_TYPE(machine)->tp_name, Py_TYPE(self)->tp_name);
    return 0;
  }

  // all went fine, pack machine and eigen-values to return
  PyObject* retval = PyTuple_New(2);

  Py_INCREF(machine);
  PyTuple_SET_ITEM(retval, 0, machine);
  Py_INCREF(eigval);
  PyTuple_SET_ITEM(retval, 1, PyBlitzArray_NUMPY_WRAP(eigval));

  return retval;

}

PyDoc_STRVAR(s_output_size_str, "output_size");
PyDoc_STRVAR(s_output_size_doc,
"o.output_size(X) -> int\n\
\n\
Calculates the maximum possible rank for the covariance\n\
matrix of ``X``, given ``X``.\n\
\n\
Returns the maximum number of non-zero eigen values that\n\
can be generated by this trainer, given some data. This\n\
number (K) depends on the size of X and is calculated as\n\
follows :math:`K=\\min{(S-1,F)}`, with :math:`S` being the\n\
number of rows in ``data`` (samples) and :math:`F` the\n\
number of columns (or features).\n\
\n\
This method should be used to setup linear machines and\n\
input vectors prior to feeding them into this trainer.\n\
\n\
");

static PyObject* PyBobLearnLinearPCATrainer_OutputSize
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"X", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* X = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist,
        &PyBlitzArray_Converter, &X)) return 0;

  auto X_ = make_safe(X); ///< auto-delete in case of problems

  if (X->ndim != 2 || X->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input array `X'", Py_TYPE(self)->tp_name);
    return 0;
  }

  // evaluates the expected rank for the output, allocate eigens value array
  auto X_bz = PyBlitzArrayCxx_AsBlitz<double,2>(X);

  return Py_BuildValue("n", self->cxx->output_size(*X_bz));

}

static PyMethodDef PyBobLearnLinearPCATrainer_methods[] = {
  {
    s_is_similar_to_str,
    (PyCFunction)PyBobLearnLinearPCATrainer_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    s_is_similar_to_doc
  },
  {
    s_train_str,
    (PyCFunction)PyBobLearnLinearPCATrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    s_train_doc
  },
  {
    s_output_size_str,
    (PyCFunction)PyBobLearnLinearPCATrainer_OutputSize,
    METH_VARARGS|METH_KEYWORDS,
    s_output_size_doc
  },
  {0} /* Sentinel */
};

PyDoc_STRVAR(s_use_svd_str, "use_svd");
PyDoc_STRVAR(s_use_svd_doc,
"This flag determines if this trainer will use the SVD method\n\
(set it to ``True``) to calculate the principal components or\n\
the Covariance method (set it to ``False``).\n\
\n\
");

static PyObject* PyBobLearnLinearPCATrainer_getUseSVD
(PyBobLearnLinearPCATrainerObject* self, void* /*closure*/) {

  if (self->cxx->getUseSVD()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static int PyBobLearnLinearPCATrainer_setUseSVD
(PyBobLearnLinearPCATrainerObject* self, PyObject* o, void* /*closure*/) {

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  if (istrue) self->cxx->setUseSVD(true);
  else self->cxx->setUseSVD(false);

  return 0;

}

#if BOB_API_VERSION >= 0x0103
PyDoc_STRVAR(s_safe_svd_str, "safe_svd");
PyDoc_STRVAR(s_safe_svd_doc,
"If the ``use_svd`` flag is enabled, this flag will indicate\n\
which LAPACK SVD function to use (``dgesvd`` if set to\n\
``True``, ``dgesdd`` otherwise). By default, this flag is set\n\
to ``False`` upon construction, which makes this trainer use\n\
the fastest possible SVD decomposition.\n\
");

static PyObject* PyBobLearnLinearPCATrainer_getSafeSVD
(PyBobLearnLinearPCATrainerObject* self, void* /*closure*/) {

  if (self->cxx->getSafeSVD()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static int PyBobLearnLinearPCATrainer_setSafeSVD
(PyBobLearnLinearPCATrainerObject* self, PyObject* o, void* /*closure*/) {

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  if (istrue) self->cxx->setSafeSVD(true);
  else self->cxx->setSafeSVD(false);

  return 0;

}
#endif

static PyGetSetDef PyBobLearnLinearPCATrainer_getseters[] = {
    {
      s_use_svd_str,
      (getter)PyBobLearnLinearPCATrainer_getUseSVD,
      (setter)PyBobLearnLinearPCATrainer_setUseSVD,
      s_use_svd_doc,
      0
    },
#     if BOB_API_VERSION >= 0x0103
    {
      s_safe_svd_str,
      (getter)PyBobLearnLinearPCATrainer_getSafeSVD,
      (setter)PyBobLearnLinearPCATrainer_setSafeSVD,
      s_safe_svd_doc,
      0
    },
#     endif
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearPCATrainer_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_pcatrainer_str,                                 /* tp_name */
    sizeof(PyBobLearnLinearPCATrainerObject),         /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearPCATrainer_delete,    /* tp_dealloc */
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
    s_pcatrainer_doc,                                 /* tp_doc */
    0,                                                /* tp_traverse */
    0,                                                /* tp_clear */
    (richcmpfunc)PyBobLearnLinearPCATrainer_RichCompare, /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    PyBobLearnLinearPCATrainer_methods,               /* tp_methods */
    0,                                                /* tp_members */
    PyBobLearnLinearPCATrainer_getseters,             /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)PyBobLearnLinearPCATrainer_init,        /* tp_init */
    0,                                                /* tp_alloc */
    0,                                                /* tp_new */
};
