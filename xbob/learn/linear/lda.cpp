/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 17:09:04 CET
 *
 * @brief Python bindings to LDA trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define XBOB_LEARN_LINEAR_MODULE
#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>
#include <bob/trainer/FisherLDATrainer.h>
#include <xbob.learn.linear/api.h>
#include <structmember.h>

/*************************************************
 * Implementation of FisherLDATrainer base class *
 *************************************************/

PyDoc_STRVAR(s_pcatrainer_str, XBOB_EXT_MODULE_PREFIX ".FisherLDATrainer");

PyDoc_STRVAR(s_pcatrainer_doc,
"FisherLDATrainer([use_pinv=False [, strip_to_rank=True]]) -> new FisherLDATrainer\n\
FisherLDATrainer(other) -> new FisherLDATrainer\n\
\n\
Trains a :py:class:`bob.machine.LinearMachine` to perform\n\
Fisher's Linear Discriminant Analysis (LDA).\n\
\n\
Objects of this class can be initialized in two ways. In the\n\
first variant, the user creates a new trainer from discrete\n\
flags indicating a couple of optional parameters:\n\
\n\
  use_pinv (bool) - defaults to ``False``\n\
     \n\
     If set to ``True``, use the pseudo-inverse to calculate\n\
     :math:`S_w^{-1} S_b` and then perform eigen value\n\
     decomposition (using LAPACK's ``dgeev``) instead of\n\
     using (the more numerically stable) LAPACK's ``dsyvgd``\n\
     to solve the generalized symmetric-definite eigenproblem\n\
     of the form :math:`S_b v=(\\lambda) S_w v`.\n\
     \n\
     .. note::\n\
       \n\
       Using the pseudo-inverse for LDA is only recommended\n\
       if you cannot make it work using the default method\n\
       (via ``dsyvg``). It is slower and requires more machine\n\
       memory to store partial values of the pseudo-inverse\n\
       and the dot product :math:`S_w^{-1} S_b`.\n\
  \n\
  strip_to_rank (bool) - defaults to ``True``\n\
     \n\
     Specifies how to calculate the final size of the\n\
     to-be-trained :py:class:`xbob.learn.linear.Machine`. The\n\
     default setting (``True``), makes the trainer return\n\
     only the K-1 eigen-values/vectors limiting the output to\n\
     the rank of :math:`S_w^{-1} S_b`. If you set this value\n\
     to ``False``, the it returns all eigen-values/vectors of\n\
     :math:`S_w^{-1} Sb`, including the ones that are supposed\n\
     to be zero.\n\
\n\
The second initialization variant allows the user to deep copy\n\
an object of the same type creating a new identical object.\n\
\n\
LDA finds the projection matrix W that allows us to linearly\n\
project the data matrix X to another (sub) space in which\n\
the between-class and within-class variances are jointly\n\
optimized: the between-class variance is maximized while the\n\
with-class is minimized. The (inverse) cost function for this\n\
criteria can be posed as the following:\n\
\n\
.. math::\n\
   \n\
   J(W) = \\frac{W^T S_b W}{W^T S_w W}\n\
\n\
where:\n\
\n\
:math:`W`\n\
   \n\
   the transformation matrix that converts X into the LD space\n\
\n\
:math:`S_b`\n\
   \n\
   the between-class scatter; it has dimensions (X.shape[0],\n\
   X.shape[0]) and is defined as\n\
   :math:`S_b = \\sum_{k=1}^K N_k (m_k-m)(m_k-m)^T`, with K\n\
   equal to the number of classes.\n\
\n\
:math:`S_w`\n\
  \n\
   the within-class scatter; it also has dimensions\n\
   (X.shape[0], X.shape[0]) and is defined as\n\
   :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`,\n\
   with K equal to the number of classes and :math:`C_k` a set\n\
   representing all samples for class k.\n\
\n\
:math:`m_k`\n\
  \n\
   the class *k* empirical mean, defined as\n\
   :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`\n\
\n\
:math:`m`\n\
  \n\
   the overall set empirical mean, defined as\n\
   :math:`m = \\frac{1}{N}\\sum_{n=1}^N x_n = \\frac{1}{N}\\sum_{k=1}^K N_k m_k`\n\
\n\
.. note::\n\
   \n\
   A scatter matrix equals the covariance matrix if we remove\n\
   the division factor.\n\
\n\
Because this cost function is convex, you can just find its\n\
maximum by solving :math:`dJ/dW = 0`. This problem can be\n\
re-formulated as finding the eigen values (:math:`\\lambda_i`)\n\
that solve the following condition:\n\
\n\
.. math::\n\
  \n\
  S_b &= \\lambda_i Sw \\text{ or} \\\\\n\
  (Sb - \\lambda_i Sw) &= 0\n\
\n\
The respective eigen vectors that correspond to the eigen\n\
values :math:`\\lambda_i` form W.\n\
\n\
");

static int PyBobLearnLinearFisherLDATrainer_init_bools
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"use_pinv", "strip_to_rank", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* use_pinv = Py_False;
  PyObject* strip_to_rank = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
        &use_pinv, &strip_to_rank)) return -1;

  int use_pinv_ = PyObject_IsTrue(use_pinv);

  if (use_pinv_ == -1) return -1;

  int strip_to_rank_ = PyObject_IsTrue(strip_to_rank);

  if (strip_to_rank_ == -1) return -1;

  try {
    self->cxx = new bob::trainer::FisherLDATrainer(use_pinv_?true:false,
        strip_to_rank_?true:false);
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

static int PyBobLearnLinearFisherLDATrainer_init_copy
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearFisherLDATrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearFisherLDATrainerObject*>(other);

  try {
    self->cxx = new bob::trainer::FisherLDATrainer(*(copy->cxx));
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

int PyBobLearnLinearFisherLDATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearFisherLDATrainer_Type));
}

static int PyBobLearnLinearFisherLDATrainer_init
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwds?PyDict_Size(kwds):0;

  switch (nargs) {

    case 0: //default initializer
      return PyBobLearnLinearFisherLDATrainer_init_bools(self, args, kwds);

    case 1:

      {

        PyObject* arg = 0; ///< borrowed (don't delete)
        if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
        else {
          PyObject* tmp = PyDict_Values(kwds);
          auto tmp_ = make_safe(tmp);
          arg = PyList_GET_ITEM(tmp, 0);
        }

        if (PyBobLearnLinearFisherLDATrainer_Check(arg)) {
          return PyBobLearnLinearFisherLDATrainer_init_copy(self, args, kwds);
        }
        else {
          return PyBobLearnLinearFisherLDATrainer_init_bools(self, args, kwds);
        }

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);

      }

      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - `%s' requires 0 or 1 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);

  }

  return -1;

}

static void PyBobLearnLinearFisherLDATrainer_delete
(PyBobLearnLinearFisherLDATrainerObject* self) {

  delete self->cxx;
  Py_TYPE(self)->tp_free((PyObject*)self);

}

static PyObject* PyBobLearnLinearFisherLDATrainer_RichCompare
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearFisherLDATrainer_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearFisherLDATrainerObject*>(other);

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
Compares this FisherLDATrainer with the ``other`` one to be\n\
approximately the same.\n\
\n\
The optional values ``r_epsilon`` and ``a_epsilon`` refer to the\n\
relative and absolute precision for the ``weights``, ``biases``\n\
and any other values internal to this machine.\n\
\n\
");

static PyObject* PyBobLearnLinearFisherLDATrainer_IsSimilarTo
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"other", "r_epsilon", "a_epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|dd", kwlist,
        &PyBobLearnLinearFisherLDATrainer_Type, &other,
        &r_epsilon, &a_epsilon)) return 0;

  auto other_ = reinterpret_cast<PyBobLearnLinearFisherLDATrainerObject*>(other);

  if (self->cxx->is_similar_to(*other_->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_train_str, "train");
PyDoc_STRVAR(s_train_doc,
"o.train(X [, machine]) -> (machine, eigen_values)\n\
\n\
Trains a given machine to perform Fisher/LDA discrimination.\n\
\n\
After this method has been called, an input machine (or one\n\
allocated internally) will have the eigen-vectors of the\n\
:math:`S_w^{-1} S_b` product, arranged by decreasing energy.\n\
Each input data set represents data from a given input class.\n\
This method also returns the eigen values allowing you to\n\
implement your own compression scheme.\n\
\n\
The user may provide or not an object of type\n\
:py:class:`xbob.learn.linear.Machine` that will be set by this\n\
method. If provided, machine should have the correct number of\n\
inputs and outputs matching, respectively, the number of columns\n\
in the input data arrays ``X`` and the output of the method\n\
:py:meth:`xbob.learn.linear.FisherLDATrainer.output_size` (see\n\
help).\n\
\n\
The value of ``X`` should be a sequence over as many 2D 64-bit\n\
floating point number arrays as classes in the problem. All\n\
arrays will be checked for conformance (identical number of\n\
columns). To accomplish this, either prepare a list with all\n\
your class observations organised in 2D arrays or pass a 3D\n\
array in which the first dimension (depth) contains as many\n\
elements as classes you want to discriminate.\n\
\n\
.. note::\n\
   \n\
   We set at most\n\
   :py:meth:`bob.trainer.FisherLDATrainer.output_size`\n\
   eigen-values and vectors on the passed machine. You can\n\
   compress the machine output further using\n\
   :py:meth:`xbob.learn.linear.Machine.resize` if necessary.\n\
\n\
");

static PyObject* PyBobLearnLinearFisherLDATrainer_Train
(PyBobLearnLinearFisherLDATrainerObject* self,
 PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"X", "machine", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* X = 0;
  PyObject* machine = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O!", kwlist,
        &X, &PyBobLearnLinearMachine_Type, &machine)) return 0;

  if (!PySequence_Check(X)) {
    PyErr_Format(PyExc_TypeError, "`%s' requires an input sequence for parameter `X', but you passed a `%s' which does not implement the sequence protocol", Py_TYPE(self)->tp_name, Py_TYPE(X)->tp_name);
    return 0;
  }

  /* Checks and converts all entries */
  std::vector<blitz::Array<double,2> > Xseq;
  std::vector<std::shared_ptr<PyBlitzArrayObject>> Xseq_;
  Py_ssize_t size = PySequence_Fast_GET_SIZE(X);

  if (size < 2) {
    PyErr_Format(PyExc_RuntimeError, "`%s' requires an input sequence for parameter `X' with at least two entries (representing two classes), but you have passed something that has only %" PY_FORMAT_SIZE_T "d entries", Py_TYPE(self)->tp_name, size);
    return 0;
  }

  Xseq.reserve(size);
  Xseq_.reserve(size);

  for (Py_ssize_t k=0; k<size; ++k) {

    PyBlitzArrayObject* bz = 0;
    PyObject* borrowed = PySequence_Fast_GET_ITEM(X, k);

    if (!PyBlitzArray_Converter(borrowed, &bz)) {
      PyErr_Format(PyExc_TypeError, "`%s' could not convert object of type `%s' at position %" PY_FORMAT_SIZE_T "d of input sequence `X' into an array - check your input", Py_TYPE(self)->tp_name, Py_TYPE(borrowed)->tp_name, k);
      return 0;
    }

    if (bz->ndim != 2 || bz->type_num != NPY_FLOAT64) {
      PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input sequence `X' (or any other object coercible to that), but at position %" PY_FORMAT_SIZE_T "d I have found an object with %" PY_FORMAT_SIZE_T "d dimensions and with type `%s' which is not compatible - check your input", Py_TYPE(self)->tp_name, k, bz->ndim, PyBlitzArray_TypenumAsString(bz->type_num));
      Py_DECREF(bz);
      return 0;
    }

    Xseq_.push_back(make_safe(bz)); ///< prevents data deletion
    Xseq.push_back(*PyBlitzArrayCxx_AsBlitz<double,2>(bz)); ///< only a view!

  }

  // evaluates the expected rank for the output, allocate eigens value array
  Py_ssize_t rank = self->cxx->output_size(Xseq);
  auto eigval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &rank);
  auto eigval_ = make_safe(eigval); ///< auto-delete in case of problems

  // allocates a new machine if that was not given by the user
  std::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(Xseq[0].extent(1), rank);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  try {
    auto eigval_bz = PyBlitzArrayCxx_AsBlitz<double,1>(reinterpret_cast<PyBlitzArrayObject*>(eigval));
    self->cxx->train(*pymac->cxx, *eigval_bz, Xseq);
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
Returns the expected size of the output (or the number of\n\
eigen-values returned) given the data.\n\
\n\
This number could be either K-1 (where K is number of classes)\n\
or the number of columns (features) in X, depending on the\n\
setting of ``strip_to_rank``.\n\
\n\
This method should be used to setup linear machines and\n\
input vectors prior to feeding them into this trainer.\n\
\n\
The value of ``X`` should be a sequence over as many 2D 64-bit\n\
floating point number arrays as classes in the problem. All\n\
arrays will be checked for conformance (identical number of\n\
columns). To accomplish this, either prepare a list with all\n\
your class observations organised in 2D arrays or pass a 3D\n\
array in which the first dimension (depth) contains as many\n\
elements as classes you want to discriminate.\n\
\n\
");

static PyObject* PyBobLearnLinearFisherLDATrainer_OutputSize
(PyBobLearnLinearFisherLDATrainerObject* self,
 PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"X", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* X = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &X)) return 0;

  if (!PySequence_Check(X)) {
    PyErr_Format(PyExc_TypeError, "`%s' requires an input sequence for parameter `X', but you passed a `%s' which does not implement the sequence protocol", Py_TYPE(self)->tp_name, Py_TYPE(X)->tp_name);
    return 0;
  }

  /* Checks and converts all entries */
  std::vector<blitz::Array<double,2> > Xseq;
  std::vector<std::shared_ptr<PyBlitzArrayObject>> Xseq_;
  Py_ssize_t size = PySequence_Fast_GET_SIZE(X);

  if (size < 2) {
    PyErr_Format(PyExc_RuntimeError, "`%s' requires an input sequence for parameter `X' with at least two entries (representing two classes), but you have passed something that has only %" PY_FORMAT_SIZE_T "d entries", Py_TYPE(self)->tp_name, size);
    return 0;
  }

  Xseq.reserve(size);
  Xseq_.reserve(size);

  for (Py_ssize_t k=0; k<size; ++k) {

    PyBlitzArrayObject* bz = 0;
    PyObject* borrowed = PySequence_Fast_GET_ITEM(X, k);

    if (!PyBlitzArray_Converter(borrowed, &bz)) {
      PyErr_Format(PyExc_TypeError, "`%s' could not convert object of type `%s' at position %" PY_FORMAT_SIZE_T "d of input sequence `X' into an array - check your input", Py_TYPE(self)->tp_name, Py_TYPE(borrowed)->tp_name, k);
      return 0;
    }

    if (bz->ndim != 2 || bz->type_num != NPY_FLOAT64) {
      PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for input sequence `X' (or any other object coercible to that), but at position %" PY_FORMAT_SIZE_T "d I have found an object with %" PY_FORMAT_SIZE_T "d dimensions and with type `%s' which is not compatible - check your input", Py_TYPE(self)->tp_name, k, bz->ndim, PyBlitzArray_TypenumAsString(bz->type_num));
      Py_DECREF(bz);
      return 0;
    }

    Xseq_.push_back(make_safe(bz)); ///< prevents data deletion
    Xseq.push_back(*PyBlitzArrayCxx_AsBlitz<double,2>(bz)); ///< only a view!

  }

  return Py_BuildValue("n", self->cxx->output_size(Xseq));

}

static PyMethodDef PyBobLearnLinearFisherLDATrainer_methods[] = {
  {
    s_is_similar_to_str,
    (PyCFunction)PyBobLearnLinearFisherLDATrainer_IsSimilarTo,
    METH_VARARGS|METH_KEYWORDS,
    s_is_similar_to_doc
  },
  {
    s_train_str,
    (PyCFunction)PyBobLearnLinearFisherLDATrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    s_train_doc
  },
  {
    s_output_size_str,
    (PyCFunction)PyBobLearnLinearFisherLDATrainer_OutputSize,
    METH_VARARGS|METH_KEYWORDS,
    s_output_size_doc
  },
  {0} /* Sentinel */
};

PyDoc_STRVAR(s_use_pinv_str, "use_pinv");
PyDoc_STRVAR(s_use_pinv_doc,
"If ``True``, use the pseudo-inverse to calculate\n\
:math:`S_w^{-1} S_b` and then perform the eigen value\n\
decomposition (using LAPACK's ``dgeev``) instead of using\n\
(the more numerically stable) LAPACK's ``dsyvgd`` to solve\n\
the generalized symmetric-definite eigenproblem of the\n\
form :math:`S_b v=(\\lambda) S_w v`\n\
");

static PyObject* PyBobLearnLinearFisherLDATrainer_getUsePinv
(PyBobLearnLinearFisherLDATrainerObject* self, void* /*closure*/) {

  if (self->cxx->getUsePseudoInverse()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static int PyBobLearnLinearFisherLDATrainer_setUsePinv
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* o, void* /*closure*/) {

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  if (istrue) self->cxx->setUsePseudoInverse(true);
  else self->cxx->setUsePseudoInverse(false);

  return 0;

}

PyDoc_STRVAR(s_strip_to_rank_str, "strip_to_rank");
PyDoc_STRVAR(s_strip_to_rank_doc,
"If the ``use_svd`` flag is enabled, this flag will indicate\n\
which LAPACK SVD function to use (``dgesvd`` if set to\n\
``True``, ``dgesdd`` otherwise). By default, this flag is set\n\
to ``False`` upon construction, which makes this trainer use\n\
the fastest possible SVD decomposition.\n\
");

static PyObject* PyBobLearnLinearFisherLDATrainer_getStripToRank
(PyBobLearnLinearFisherLDATrainerObject* self, void* /*closure*/) {

  if (self->cxx->getStripToRank()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static int PyBobLearnLinearFisherLDATrainer_setStripToRank
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* o, void* /*closure*/) {

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  if (istrue) self->cxx->setStripToRank(true);
  else self->cxx->setStripToRank(false);

  return 0;

}

static PyGetSetDef PyBobLearnLinearFisherLDATrainer_getseters[] = {
    {
      s_use_pinv_str,
      (getter)PyBobLearnLinearFisherLDATrainer_getUsePinv,
      (setter)PyBobLearnLinearFisherLDATrainer_setUsePinv,
      s_use_pinv_doc,
      0
    },
    {
      s_strip_to_rank_str,
      (getter)PyBobLearnLinearFisherLDATrainer_getStripToRank,
      (setter)PyBobLearnLinearFisherLDATrainer_setStripToRank,
      s_strip_to_rank_doc,
      0
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobLearnLinearFisherLDATrainer_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_pcatrainer_str,                                 /* tp_name */
    sizeof(PyBobLearnLinearFisherLDATrainerObject),   /* tp_basicsize */
    0,                                                /* tp_itemsize */
    (destructor)PyBobLearnLinearFisherLDATrainer_delete, /* tp_dealloc */
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
    (richcmpfunc)PyBobLearnLinearFisherLDATrainer_RichCompare, /* tp_richcompare */
    0,                                                /* tp_weaklistoffset */
    0,                                                /* tp_iter */
    0,                                                /* tp_iternext */
    PyBobLearnLinearFisherLDATrainer_methods,         /* tp_methods */
    0,                                                /* tp_members */
    PyBobLearnLinearFisherLDATrainer_getseters,       /* tp_getset */
    0,                                                /* tp_base */
    0,                                                /* tp_dict */
    0,                                                /* tp_descr_get */
    0,                                                /* tp_descr_set */
    0,                                                /* tp_dictoffset */
    (initproc)PyBobLearnLinearFisherLDATrainer_init,  /* tp_init */
    0,                                                /* tp_alloc */
    0,                                                /* tp_new */
};
