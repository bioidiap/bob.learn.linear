/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 14:27:40 CET
 *
 * @brief Python bindings to PCA trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.learn.linear/api.h>
#include <bob.extension/documentation.h>
#include <structmember.h>

/*******************************************
 * Implementation of PCATrainer base class *
 *******************************************/
static auto PCA_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".PCATrainer",
  "Sets a linear machine to perform the Principal Component Analysis (PCA; a.k.a. Karhunen-Loeve Transform -- KLT) on a given dataset using either Singular Value Decomposition (SVD, *the default*) or the Covariance Matrix Method",
  "The training stage will place the resulting principal components in the linear machine and set it up to extract the variable means automatically. "
  "As an option, you may preset the trainer so that the normalization performed by the resulting linear machine also divides the variables by the standard deviation of each variable ensemble. "
  "The principal components correspond the direction of the data in which its points are maximally spread.\n\n"
  "Computing these principal components is equivalent to computing the eigen-vectors :math:`U` for the covariance matrix :math:`\\Sigma` extracted from the data matrix :math:`X`. "
  "The covariance matrix for the data is computed using the equation below:\n\n"
  ".. math::\n\n"
  "   \\Sigma &= \\frac{((X-\\mu_X)^T(X-\\mu_X))}{m-1} \\text{ with}\\\\\n"
  "   \\mu_X  &= \\sum_i^N x_i\n\n"
  "where :math:`m` is the number of rows in :math:`X` (that is, the number of samples).\n\n"
  "Once you are in possession of :math:`\\Sigma`, it suffices to compute the eigen-vectors :math:`U`, solving the linear equation:\n\n"
  ".. math:: (\\Sigma - e I) U = 0\n\n"
  "In this trainer, we make use of LAPACK's ``dsyevd`` to solve the above equation, if you choose to use the Covariance Method for extracting the principal components of your data matrix :math:`X`.\n\n"
  "**By default** though, this class will perform PC extraction using Singular Value Decomposition (SVD). "
  "SVD is a factorization technique that allows for the decomposition of a matrix :math:`X`, with size (m,n) into 3 other matrices in this way:\n\n"
  ".. math:: X = U S V^*\n\n"
  "where:\n\n"
  ":math:`U`\n\n  unitary matrix of size (m,m) - a.k.a., left singular vectors of :math:`X`\n\n"
  ":math:`S`\n\n  rectangular diagonal matrix with nonnegative real numbers, size (m,n)\n\n"
  ":math:`V^*`\n\n  (the conjugate transpose of :math:`V`) unitary matrix of size (n,n), a.k.a. right singular vectors of :math:`X`\n\n"
  "We can use this property to avoid the computation of the covariance matrix of the data matrix :math:`X`, if we note the following:\n\n"
  ".. math::\n\n"
  "   X &= U S V^* \\text{ , so} \\\\\n"
  "   XX^T &= U S V^* V S U^*\\\\\n"
  "   XX^T &= U S^2 U^*\n\n"
  "If :math:`X` has zero mean, we can conclude by inspection that the :math:`U` matrix obtained by SVD contains the eigen-vectors of the covariance matrix of :math:`X` (:math:`XX^T`) and :math:`S^2/(m-1)` corresponds to its eigen values.\n\n"
  ".. note:: Our implementation uses LAPACK's ``dgesdd`` to compute the solution to this linear equation.\n\n"
  "The corresponding :py:class:`bob.learn.linear.Machine` and returned eigen-values of :math:`\\Sigma`, are pre-sorted in descending order (the first eigen-vector - or column - of the weight matrix in the :py:class:`bob.learn.linear.Machine` corresponds to the highest eigen-value obtained).\n\n"
  ".. note::\n\n"
  "   One question you should pose yourself is which of the methods to choose.\n"
  "   Here is some advice: you should prefer the covariance method over SVD when the number of samples (rows of :math:`X`) is greater than the number of features (columns of :math:`X`).\n"
  "   It provides a faster execution path in that case.\n"
  "   Otherwise, use the **default** SVD method.\n\n"
  "References:\n\n"
  "1. Eigenfaces for Recognition, Turk & Pentland, Journal of Cognitive Neuroscience (1991) Volume: 3, Issue: 1, Publisher: MIT Press, Pages: 71-86\n"
  "2. http://en.wikipedia.org/wiki/Singular_value_decomposition\n"
  "3. http://en.wikipedia.org/wiki/Principal_component_analysis\n"
  "4. http://www.netlib.org/lapack/double/dsyevd.f\n"
  "5. http://www.netlib.org/lapack/double/dgesdd.f\n\n"
)
.add_constructor(bob::extension::FunctionDoc(
  "PCATrainer",
  "Constructs a new PCA trainer",
  "There are two initializers for objects of this class. "
  "In the first variant, the user can pass a flag indicating if the trainer should use SVD (default) or the covariance method for PCA extraction. "
  "The second initialization form copy constructs a new trainer from an existing one."
)
.add_prototype("[use_svd]","")
.add_prototype("other", "")
.add_parameter("use_svd", "bool", "[Default: ``True``] Use SVD for computing the PCA?")
.add_parameter("other", ":py:class:`PCATrainer`", "The trainer to copy-construct")
);
static int PyBobLearnLinearPCATrainer_init_bool
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = PCA_doc.kwlist(0);

  PyObject* use_svd = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &use_svd)) return -1;

  int use_svd_ = PyObject_IsTrue(use_svd);
  if (use_svd_ == -1) return -1; //error on conversion

  self->cxx = new bob::learn::linear::PCATrainer(use_svd_);

  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearPCATrainer_init_copy
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  static char** kwlist = PCA_doc.kwlist(1);

  PyBobLearnLinearPCATrainerObject* other;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearPCATrainer_Type, &other)) return -1;

  self->cxx = new bob::learn::linear::PCATrainer(*other->cxx);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBobLearnLinearPCATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearPCATrainer_Type));
}

static int PyBobLearnLinearPCATrainer_init
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {

  PyObject* arg = 0; ///< borrowed (don't delete)
  if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
  else {
    if (!kwds) return PyBobLearnLinearPCATrainer_init_bool(self, args, kwds);
    PyObject* tmp = PyDict_Values(kwds);
    auto tmp_ = make_safe(tmp);
    arg = PyList_GET_ITEM(tmp, 0);
  }

  if (PyBobLearnLinearPCATrainer_Check(arg)) {
    return PyBobLearnLinearPCATrainer_init_copy(self, args, kwds);
  }

  return PyBobLearnLinearPCATrainer_init_bool(self, args, kwds);

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

static auto train = bob::extension::FunctionDoc(
  "train",
  "Trains a linear machine to perform the PCA (aka. KLT)",
  "The resulting machine will have the same number of inputs as columns in ``X`` and :math:`K` eigen-vectors, where :math:`K=\\min{(S-1,F)}`, with :math:`S` being the number of rows in ``X`` (samples) and :math:`F` the number of columns (or features). "
  "The vectors are arranged by decreasing eigen-value automatically -- there is no need to sort the results.\n\n"
  "The user may provide or not an object of type :py:class:`bob.learn.linear.Machine` that will be set by this method. "
  "If provided, machine should have the correct number of inputs and outputs matching, respectively, the number of columns in the input data array ``X`` and the output of the method :py:meth:`output_size`.\n\n"
  "The input data matrix ``X`` should correspond to a 64-bit floating point array organized in such a way that every row corresponds to a new observation of the phenomena (i.e., a new sample) and every column corresponds to a different feature.\n\n"
  "This method returns a tuple consisting of the trained machine and a 1D 64-bit floating point array containing the eigen-values calculated while computing the KLT. "
  "The eigen-value ordering matches that of eigen-vectors set in the machine.",
  true
)
.add_prototype("X, [machine]", "machine, eigen_values")
.add_parameter("X", "array_like(2D, floats)", "The input data to train on")
.add_parameter("machine", ":py:class:`bob.learn.linear.Machine`", "The machine to be trained; this machine will be returned by this function")
.add_return("machine", ":py:class:`bob.learn.linear.Machine`", "The machine that has been trained; if given, identical to the ``machine`` parameter")
.add_return("eigen_values", "array_like(1D, floats)", "The eigen-values of the PCA projection.")
;
static PyObject* PyBobLearnLinearPCATrainer_Train
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = train.kwlist();

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
  auto eigval = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &rank));
  auto eigval_ = make_safe(eigval); ///< auto-delete in case of problems

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(X_bz->extent(1), rank);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  auto eigval_bz = PyBlitzArrayCxx_AsBlitz<double,1>(eigval);
  self->cxx->train(*pymac->cxx, *eigval_bz, *X_bz);

  // all went fine, pack machine and eigen-values to return
  return Py_BuildValue("ON", machine, PyBlitzArray_AsNumpyArray(eigval, 0));
BOB_CATCH_MEMBER("train", 0)
}

static auto output_size = bob::extension::FunctionDoc(
  "output_size",
  "Calculates the maximum possible rank for the covariance matrix of the given ``X``",
  "Returns the maximum number of non-zero eigen values that can be generated by this trainer, given ``X``. "
  "This number (K) depends on the size of X and is calculated as follows :math:`K=\\min{(S-1,F)}`, with :math:`S` being the number of rows in ``data`` (samples) and :math:`F` the number of columns (or features).\n\n"
  "This method should be used to setup linear machines and input vectors prior to feeding them into the :py:meth:`train` function.",
  true
)
.add_prototype("X","size")
.add_parameter("X", "array_like(2D, floats)", "The input data that should be trained on")
.add_return("size", "int", "The number of eigen-vectors/values that will be created in a call to :py:meth:`train`, given the same input data ``X``")
;
static PyObject* PyBobLearnLinearPCATrainer_OutputSize
(PyBobLearnLinearPCATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = output_size.kwlist();

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
BOB_CATCH_MEMBER("output_size", 0)
}

static PyMethodDef PyBobLearnLinearPCATrainer_methods[] = {
  {
    train.name(),
    (PyCFunction)PyBobLearnLinearPCATrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    train.doc()
  },
  {
    output_size.name(),
    (PyCFunction)PyBobLearnLinearPCATrainer_OutputSize,
    METH_VARARGS|METH_KEYWORDS,
    output_size.doc()
  },
  {0} /* Sentinel */
};

static auto use_svd = bob::extension::VariableDoc(
  "use_svd",
  "bool",
  "Use the SVD to compute PCA?",
  "This flag determines if this trainer will use the SVD method (set it to ``True``) to calculate the principal components or the Covariance method (set it to ``False``)."
);
static PyObject* PyBobLearnLinearPCATrainer_getUseSVD
(PyBobLearnLinearPCATrainerObject* self, void* /*closure*/) {
BOB_TRY
  if (self->cxx->getUseSVD()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER("use_svd", 0)
}

static int PyBobLearnLinearPCATrainer_setUseSVD
(PyBobLearnLinearPCATrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;
  self->cxx->setUseSVD(istrue);
  return 0;
BOB_CATCH_MEMBER("use_svd", -1)
}

static auto safe_svd = bob::extension::VariableDoc(
  "safe_svd",
  "bool",
  "Use the safe LAPACK SVD function?",
  "If the :py:attr:`use_svd` flag is enabled, this flag will indicate which LAPACK SVD function to use (``dgesvd`` if set to ``True``, ``dgesdd`` otherwise). "
  "By default, this flag is set to ``False`` upon construction, which makes this trainer use the fastest possible SVD decomposition."
);
static PyObject* PyBobLearnLinearPCATrainer_getSafeSVD
(PyBobLearnLinearPCATrainerObject* self, void* /*closure*/) {
BOB_TRY
  if (self->cxx->getSafeSVD()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER("safe_svd", 0)
}

static int PyBobLearnLinearPCATrainer_setSafeSVD
(PyBobLearnLinearPCATrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;
  self->cxx->setSafeSVD(istrue);
  return 0;
BOB_CATCH_MEMBER("safe_svd", -1)
}

static PyGetSetDef PyBobLearnLinearPCATrainer_getseters[] = {
    {
      use_svd.name(),
      (getter)PyBobLearnLinearPCATrainer_getUseSVD,
      (setter)PyBobLearnLinearPCATrainer_setUseSVD,
      use_svd.doc(),
      0
    },
    {
      safe_svd.name(),
      (getter)PyBobLearnLinearPCATrainer_getSafeSVD,
      (setter)PyBobLearnLinearPCATrainer_setSafeSVD,
      safe_svd.doc(),
      0
    },
    {0}  /* Sentinel */
};


// PCA Trainer
PyTypeObject PyBobLearnLinearPCATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearPCA(PyObject* module)
{
  // PCA Trainer
  PyBobLearnLinearPCATrainer_Type.tp_name = PCA_doc.name();
  PyBobLearnLinearPCATrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearPCATrainerObject);
  PyBobLearnLinearPCATrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearPCATrainer_Type.tp_doc = PCA_doc.doc();

  // set the functions
  PyBobLearnLinearPCATrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearPCATrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearPCATrainer_init);
  PyBobLearnLinearPCATrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearPCATrainer_delete);
  PyBobLearnLinearPCATrainer_Type.tp_methods = PyBobLearnLinearPCATrainer_methods;
  PyBobLearnLinearPCATrainer_Type.tp_getset = PyBobLearnLinearPCATrainer_getseters;
  PyBobLearnLinearPCATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearPCATrainer_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearPCATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearPCATrainer_Type);
  return PyModule_AddObject(module, "PCATrainer", (PyObject*)&PyBobLearnLinearPCATrainer_Type) >= 0;
}
