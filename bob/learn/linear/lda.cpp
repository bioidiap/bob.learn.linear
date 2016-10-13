/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jan 2014 17:09:04 CET
 *
 * @brief Python bindings to LDA trainers
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.learn.linear/api.h>
#include <structmember.h>
#include <bob.extension/documentation.h>

/*************************************************
 * Implementation of FisherLDATrainer base class *
 *************************************************/

static auto LDA_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".FisherLDATrainer",
  "Trains a :py:class:`bob.learn.linear.Machine` to perform Fisher's Linear Discriminant Analysis (LDA).",
  "LDA finds the projection matrix W that allows us to linearly project the data matrix X to another (sub) space in which the between-class and within-class variances are jointly optimized: the between-class variance is maximized while the with-class is minimized. "
  "The (inverse) cost function for this criteria can be posed as the following:\n\n"
  ".. math::\n\n"
  "   J(W) = \\frac{W^T S_b W}{W^T S_w W}\n\n"
  "where:\n\n"
  ":math:`W`\n\n  the transformation matrix that converts X into the LD space\n\n"
  ":math:`S_b`\n\n  the between-class scatter; it has dimensions (X.shape[0], X.shape[0]) and is defined as :math:`S_b = \\sum_{k=1}^K N_k (m_k-m)(m_k-m)^T`, with :math:`K` equal to the number of classes.\n\n"
  ":math:`S_w`\n\n  the within-class scatter; it also has dimensions (X.shape[0], X.shape[0]) and is defined as :math:`S_w = \\sum_{k=1}^K \\sum_{n \\in C_k} (x_n-m_k)(x_n-m_k)^T`, with :math:`K` equal to the number of classes and :math:`C_k` a set representing all samples for class :math:`k`.\n\n"
  ":math:`m_k`\n\n  the class *k* empirical mean, defined as :math:`m_k = \\frac{1}{N_k}\\sum_{n \\in C_k} x_n`\n\n"
  ":math:`m`\n\n  the overall set empirical mean, defined as :math:`m = \\frac{1}{N}\\sum_{n=1}^N x_n = \\frac{1}{N}\\sum_{k=1}^K N_k m_k`\n\n"
  ".. note::  A scatter matrix equals the covariance matrix if we remove the division factor.\n\n"
  "Because this cost function is convex, you can just find its maximum by solving :math:`dJ/dW = 0`. "
  "This problem can be re-formulated as finding the eigen-values (:math:`\\lambda_i`) that solve the following condition:\n\n"
  ".. math::\n\n"
  "   S_b &= \\lambda_i Sw \\text{ or} \\\\\n"
  "  (Sb - \\lambda_i Sw) &= 0\n\n"
  "The respective eigen-vectors that correspond to the eigen-values :math:`\\lambda_i` form W."
).add_constructor(bob::extension::FunctionDoc(
  "FisherLDATrainer",
  "Constructs a new FisherLDATrainer",
  "Objects of this class can be initialized in two ways. "
  "In the first variant, the user creates a new trainer from discrete flags indicating a couple of optional parameters. "
  "If ``use_pinv`` is set to ``True``, use the pseudo-inverse to calculate :math:`S_w^{-1} S_b` and then perform eigen value decomposition (using LAPACK's ``dgeev``) instead of using (the more numerically stable) LAPACK's ``dsyvgd`` to solve the generalized symmetric-definite eigen-problem of the form :math:`S_b v=(\\lambda) S_w v`.\n\n"
  ".. note::\n\n"
  "   Using the pseudo-inverse for LDA is only recommended if you cannot make it work using the default method (via ``dsyvg``).\n"
  "   It is slower and requires more machine memory to store partial values of the pseudo-inverse and the dot product :math:`S_w^{-1} S_b`.\n\n"
  "``strip_to_rank`` specifies how to calculate the final size of the to-be-trained :py:class:`bob.learn.linear.Machine`. "
  "The default setting (``True``), makes the trainer return only the K-1 eigen-values/vectors limiting the output to the rank of :math:`S_w^{-1} S_b`. "
  "If you set this value to ``False``, the it returns all eigen-values/vectors of :math:`S_w^{-1} Sb`, including the ones that are supposed to be zero.\n\n"
  "The second initialization variant allows the user to deep copy an object of the same type creating a new identical object."
)
.add_prototype("[use_pinv, strip_to_rank]", "")
.add_prototype("other", "")
.add_parameter("use_pinv", "bool", "[Default: ``False``] use the pseudo-inverse to calculate :math:`S_w^{-1} S_b`?")
.add_parameter("strip_to_rank", "bool", "[Default: ``True``] return only the non-zero eigen-values/vectors")
.add_parameter("other", ":py:class:`FisherLDATrainer`", "The trainer to copy-construct")
);

static int PyBobLearnLinearFisherLDATrainer_init_bools
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = LDA_doc.kwlist(0);

  PyObject* use_pinv = Py_False;
  PyObject* strip_to_rank = Py_True;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist,
        &use_pinv, &strip_to_rank)) return -1;

  int use_pinv_ = PyObject_IsTrue(use_pinv);
  if (use_pinv_ == -1) return -1;

  int strip_to_rank_ = PyObject_IsTrue(strip_to_rank);
  if (strip_to_rank_ == -1) return -1;

  self->cxx = new bob::learn::linear::FisherLDATrainer(use_pinv_, strip_to_rank_);
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static int PyBobLearnLinearFisherLDATrainer_init_copy
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = LDA_doc.kwlist(1);

  PyObject* other = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
        &PyBobLearnLinearFisherLDATrainer_Type, &other)) return -1;

  auto copy = reinterpret_cast<PyBobLearnLinearFisherLDATrainerObject*>(other);

  self->cxx = new bob::learn::linear::FisherLDATrainer(*(copy->cxx));

  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

int PyBobLearnLinearFisherLDATrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearFisherLDATrainer_Type));
}

static int PyBobLearnLinearFisherLDATrainer_init
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 0: //default initializer
    case 2: //two bools
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
        } else {
          return PyBobLearnLinearFisherLDATrainer_init_bools(self, args, kwds);
        }
        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);
      }
      break;
    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - `%s' requires 0 to 2 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);

  } // switch
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

static auto train = bob::extension::FunctionDoc(
  "train",
  "Trains a given machine to perform Fisher/LDA discrimination",
  "After this method has been called, an input ``machine`` (or one allocated internally) will have the eigen-vectors of the :math:`S_w^{-1} S_b` product, arranged by decreasing energy. "
  "Each input data set represents data from a given input class. "
  "This method also returns the eigen-values allowing you to implement your own compression scheme.\n\n"
  "The user may provide or not an object of type :py:class:`bob.learn.linear.Machine` that will be set by this method. "
  "If provided, machine should have the correct number of inputs and outputs matching, respectively, the number of columns in the input data arrays ``X`` and the output of the method :py:meth:`output_size`.\n\n"
  "The value of ``X`` should be a sequence over as many 2D 64-bit floating point number arrays as classes in the problem. "
  "All arrays will be checked for conformance (identical number of columns). "
  "To accomplish this, either prepare a list with all your class observations organized in 2D arrays or pass a 3D array in which the first dimension (depth) contains as many elements as classes you want to discriminate.\n\n"
  ".. note::\n\n"
  "   We set at most :py:meth:`output_size` eigen-values and vectors on the passed machine.\n"
  "   You can compress the machine output further using :py:meth:`Machine.resize` if necessary.",
  true
)
.add_prototype("X, [machine]", "machine, eigen_values")
.add_parameter("X", "[array_like(2D, floats)] or array_like(3D, floats)", "The input data, separated to contain the training data per class in the first dimension")
.add_parameter("machine", ":py:class:`bob.learn.linear.Machine`", "The machine to be trained; this machine will be returned by this function")
.add_return("machine", ":py:class:`bob.learn.linear.Machine`", "The machine that has been trained; if given, identical to the ``machine`` parameter")
.add_return("eigen_values", "array_like(1D, floats)", "The eigen-values of the LDA projection.")
;
static PyObject* PyBobLearnLinearFisherLDATrainer_Train
(PyBobLearnLinearFisherLDATrainerObject* self,
 PyObject* args, PyObject* kwds) {

BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = train.kwlist();

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

  // evaluates the expected rank for the output, allocate eigens value array
  Py_ssize_t rank = self->cxx->output_size(Xseq);
  auto eigval = reinterpret_cast<PyBlitzArrayObject*>(PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &rank));
  auto eigval_ = make_safe(eigval); ///< auto-delete in case of problems

  // allocates a new machine if that was not given by the user
  boost::shared_ptr<PyObject> machine_;
  if (!machine) {
    machine = PyBobLearnLinearMachine_NewFromSize(Xseq[0].extent(1), rank);
    machine_ = make_safe(machine); ///< auto-delete in case of problems
  }

  auto pymac = reinterpret_cast<PyBobLearnLinearMachineObject*>(machine);

  auto eigval_bz = PyBlitzArrayCxx_AsBlitz<double,1>(eigval);
  self->cxx->train(*pymac->cxx, *eigval_bz, Xseq);

  // all went fine, pack machine and eigen-values to return
  return Py_BuildValue("ON", machine, PyBlitzArray_AsNumpyArray(eigval, 0));
BOB_CATCH_FUNCTION("train", 0)
}

static auto output_size = bob::extension::FunctionDoc(
  "output_size",
  "Returns the expected size of the output (or the number of eigen-values returned) given the data",
  "This number could be either :math:`K-1` (where :math:`K` is number of classes) or the number of columns (features) in ``X``, depending on the setting of :py:attr:`strip_to_rank`. "
  "This method should be used to setup linear machines and input vectors prior to feeding them into this trainer.\n\n"
  "The value of ``X`` should be a sequence over as many 2D 64-bit floating point number arrays as classes in the problem. "
  "All arrays will be checked for conformance (identical number of columns). "
  "To accomplish this, either prepare a list with all your class observations organized in 2D arrays or pass a 3D array in which the first dimension (depth) contains as many elements as classes you want to discriminate.",
  true
)
.add_prototype("X","size")
.add_parameter("X", "[array_like(2D, floats)] or array_like(3D, floats)", "The input data, separated to contain the training data per class in the first dimension")
.add_return("size", "int", "The number of eigen-vectors/values that will be created in a call to :py:meth:`train`, given the same input data ``X``")
;
static PyObject* PyBobLearnLinearFisherLDATrainer_OutputSize
(PyBobLearnLinearFisherLDATrainerObject* self,
 PyObject* args, PyObject* kwds) {

BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = output_size.kwlist();

  PyObject* X = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &X)) return 0;

  if (!PySequence_Check(X)) {
    PyErr_Format(PyExc_TypeError, "`%s' requires an input sequence for parameter `X', but you passed a `%s' which does not implement the sequence protocol", Py_TYPE(self)->tp_name, Py_TYPE(X)->tp_name);
    return 0;
  }

  /* Checks and converts all entries */
  std::vector<blitz::Array<double,2> > Xseq;
  std::vector<boost::shared_ptr<PyBlitzArrayObject>> Xseq_;
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
BOB_CATCH_MEMBER("output_size", 0)
}

static PyMethodDef PyBobLearnLinearFisherLDATrainer_methods[] = {
  {
    train.name(),
    (PyCFunction)PyBobLearnLinearFisherLDATrainer_Train,
    METH_VARARGS|METH_KEYWORDS,
    train.doc()
  },
  {
    output_size.name(),
    (PyCFunction)PyBobLearnLinearFisherLDATrainer_OutputSize,
    METH_VARARGS|METH_KEYWORDS,
    output_size.doc()
  },
  {0} /* Sentinel */
};

static auto use_pinv = bob::extension::VariableDoc(
  "use_pinv",
  "bool",
  "Use the pseudo-inverse?",
  "If ``True``, use the pseudo-inverse to calculate :math:`S_w^{-1} S_b` and then perform the eigen value decomposition (using LAPACK's ``dgeev``) instead of using (the more numerically stable) LAPACK's ``dsyvgd`` to solve the generalized symmetric-definite eigen-problem of the form :math:`S_b v=(\\lambda) S_w v`."
);
static PyObject* PyBobLearnLinearFisherLDATrainer_getUsePinv
(PyBobLearnLinearFisherLDATrainerObject* self, void* /*closure*/) {
BOB_TRY
  if (self->cxx->getUsePseudoInverse()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER("use_pinv", 0)
}

static int PyBobLearnLinearFisherLDATrainer_setUsePinv
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  self->cxx->setUsePseudoInverse(istrue);

  return 0;
BOB_CATCH_MEMBER("use_pinv", -1)
}

static auto strip_to_rank = bob::extension::VariableDoc(
  "strip_to_rank",
  "bool",
  "Only return the non-zero eigen-values/vectors?",
  "If ``True``, strip the resulting LDA projection matrix to keep only the eigen-vectors with non-zero eigenvalues. "
  "Otherwise the full projection matrix is returned."
);
static PyObject* PyBobLearnLinearFisherLDATrainer_getStripToRank
(PyBobLearnLinearFisherLDATrainerObject* self, void* /*closure*/) {
BOB_TRY
  if (self->cxx->getStripToRank()) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
BOB_CATCH_MEMBER("strip_to_rank", 0)
}

static int PyBobLearnLinearFisherLDATrainer_setStripToRank
(PyBobLearnLinearFisherLDATrainerObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY

  int istrue = PyObject_IsTrue(o);

  if (istrue == -1) return -1;

  self->cxx->setStripToRank(istrue);

  return 0;
BOB_CATCH_MEMBER("strip_to_rank", -1)
}

static PyGetSetDef PyBobLearnLinearFisherLDATrainer_getseters[] = {
    {
      use_pinv.name(),
      (getter)PyBobLearnLinearFisherLDATrainer_getUsePinv,
      (setter)PyBobLearnLinearFisherLDATrainer_setUsePinv,
      use_pinv.doc(),
      0
    },
    {
      strip_to_rank.name(),
      (getter)PyBobLearnLinearFisherLDATrainer_getStripToRank,
      (setter)PyBobLearnLinearFisherLDATrainer_setStripToRank,
      strip_to_rank.doc(),
      0
    },
    {0}  /* Sentinel */
};

// LDA Trainer
PyTypeObject PyBobLearnLinearFisherLDATrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobLearnLinearLDA(PyObject* module)
{
  // LDA Trainer
  PyBobLearnLinearFisherLDATrainer_Type.tp_name = LDA_doc.name();
  PyBobLearnLinearFisherLDATrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearFisherLDATrainerObject);
  PyBobLearnLinearFisherLDATrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearFisherLDATrainer_Type.tp_doc = LDA_doc.doc();

  // set the functions
  PyBobLearnLinearFisherLDATrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearFisherLDATrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearFisherLDATrainer_init);
  PyBobLearnLinearFisherLDATrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearFisherLDATrainer_delete);
  PyBobLearnLinearFisherLDATrainer_Type.tp_methods = PyBobLearnLinearFisherLDATrainer_methods;
  PyBobLearnLinearFisherLDATrainer_Type.tp_getset = PyBobLearnLinearFisherLDATrainer_getseters;
  PyBobLearnLinearFisherLDATrainer_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearFisherLDATrainer_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearFisherLDATrainer_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearFisherLDATrainer_Type);
  return PyModule_AddObject(module, "FisherLDATrainer", (PyObject*)&PyBobLearnLinearFisherLDATrainer_Type) >= 0;
}
