/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jun  4 17:44:24 CEST 2014
 *
 * @brief Bindings for a Gabor wavelet transform
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.learn.linear/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto BICMachine_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".BICMachine",
    "This machine is designed to classify image difference vectors to be either intrapersonal or extrapersonal",
    "There are two possible implementations of the BIC:\n\n"
    "* 'The Bayesian Intrapersonal/Extrapersonal Classifier' from Teixeira [Teixeira2003]_. "
    "A full projection of the data is performed. No prior for the classes has to be selected.\n"
    "* 'Face Detection and Recognition using Maximum Likelihood Classifiers on Gabor Graphs' from Guenther and Wuertz [Guenther2009]_."
    "Only mean and variance of the difference vectors are calculated. "
    "There is no subspace truncation and no priors.\n\n"
    "What kind of machine is used is dependent on the way, this class is trained via the :py:class:`bob.learn.linear.BICTrainer`.\n\n"
    ".. [Teixeira2003] **Marcio Luis Teixeira**. *The Bayesian intrapersonal/extrapersonal classifier*, Colorado State University, 2003.\n"
    ".. [Guenther2009] **Manuel Guenther and Rolf P. Wuertz**. *Face detection and recognition using maximum likelihood classifiers on Gabor graphs*, International Journal of Pattern Recognition and Artificial Intelligence, 23(3):433-461, 2009."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a BIC Machine",
    0,
    true
  )
  .add_prototype("[use_DFFS]", "")
  .add_prototype("bic", "")
  .add_prototype("hdf5", "")
  .add_parameter("use_DFFS", "bool", "[default: ``False``] Use the *Distance From Feature Space* measure as described in [Teixeira2003]_")
  .add_parameter("bic", ":py:class:`bob.learn.linear.BICMachine`", "Another machine to copy")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for reading")
);

static int PyBobLearnLinearBICMachine_init(PyBobLearnLinearBICMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist0 = BICMachine_doc.kwlist(0);
  char** kwlist1 = BICMachine_doc.kwlist(1);
  char** kwlist2 = BICMachine_doc.kwlist(2);

  // two ways to call
  PyObject* k1 = Py_BuildValue("s", kwlist1[0]),* k2 = Py_BuildValue("s", kwlist2[0]);
  auto k1_ = make_safe(k1), k2_ = make_safe(k2);
  if (
    (kwargs && PyDict_Contains(kwargs, k2)) ||
    (args && PyTuple_Size(args) == 1 && PyBobIoHDF5File_Check(PyTuple_GetItem(args, 0)))
  ){
    // HDF5
    PyBobIoHDF5FileObject* hdf5;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,"O&", kwlist2, &PyBobIoHDF5File_Converter, &hdf5)) return -1;

    auto hdf5_ = make_safe(hdf5);

    self->cxx.reset(new bob::learn::linear::BICMachine(*hdf5->f));
  } else if (
    (kwargs && PyDict_Contains(kwargs, k1)) ||
    (args && PyTuple_Size(args) == 1 && PyBobLearnLinearBICMachine_Check(PyTuple_GetItem(args, 0)))
  ){
    // copy construction
    PyBobLearnLinearBICMachineObject* bic;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,"O!", kwlist1, &PyBobLearnLinearBICMachine_Type, &bic)) return -1;

    self->cxx.reset(new bob::learn::linear::BICMachine(*bic->cxx));
  } else {
    // empty constructor
    PyObject* dffs = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist0, &dffs)) return -1;
    self->cxx.reset(new bob::learn::linear::BICMachine(dffs && PyObject_IsTrue(dffs)));
  }
  return 0;
BOB_CATCH_MEMBER("constructor",-1)
}

static void PyBobLearnLinearBICMachine_delete(PyBobLearnLinearBICMachineObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobLearnLinearBICMachine_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearBICMachine_Type));
}

static PyObject* PyBobLearnLinearBICMachine_RichCompare(PyBobLearnLinearBICMachineObject* self, PyObject* other, int op) {

  if (!PyBobLearnLinearBICMachine_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobLearnLinearBICMachineObject*>(other);

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



static auto BICTrainer_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".BICTrainer",
    "A trainer for a :py:class:`bob.learn.linear.BICMachine`",
    "It trains either a BIC model (including projection matrix and eigenvalues) [Teixeira2003]_ or an IEC model (containing mean and variance only) [Guenther2009]_. "
    "See :py:class:`bob.learn.linear.BICMachine` for more details."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Creates a BIC Trainer",
    "There are two ways of creating a BIC trainer. "
    "When you specify the ``intra_dim`` and ``extra_dim`` subspaces, a BIC model will be created, otherwise an IEC model is created.",
    true
  )
  .add_prototype("", "")
  .add_prototype("intra_dim, extra_dim", "")
  .add_parameter("intra_dim", "int", "The subspace dimensionality of the intrapersonal class")
  .add_parameter("extra_dim", "int", "The subspace dimensionality of the extrapersonal class")
);

static int PyBobLearnLinearBICTrainer_init(PyBobLearnLinearBICTrainerObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = BICTrainer_doc.kwlist(1);

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  switch (nargs){
    case 0:{
      // IEC model, no parameters
      self->cxx.reset(new bob::learn::linear::BICTrainer());
      return 0;
    }
    case 2:{
      // BIC model, two parameters
      int in, ex;
      if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &in, &ex)) return -1;
      self->cxx.reset(new bob::learn::linear::BICTrainer(in,ex));
      return 0;
    }
    default:
      BICTrainer_doc.print_usage();
      PyErr_Format(PyExc_RuntimeError, "`%s' constructor called with an unsupported number of arguments", Py_TYPE(self)->tp_name);
      return -1;
  }
BOB_CATCH_MEMBER("constructor", -1)
}

static void PyBobLearnLinearBICTrainer_delete(PyBobLearnLinearBICTrainerObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobLearnLinearBICTrainer_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobLearnLinearBICTrainer_Type));
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto dffs_doc = bob::extension::VariableDoc(
  "use_DFFS",
  "bool",
  "Use the Distance From Feature Space during forwarding?"
);
PyObject* PyBobLearnLinearBICMachine_getDFFS(PyBobLearnLinearBICMachineObject* self, void*){
BOB_TRY
  if (self->cxx->use_DFFS()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
BOB_CATCH_MEMBER("use_DFFS", 0)
}
int PyBobLearnLinearBICMachine_setDFFS(PyBobLearnLinearBICMachineObject* self, PyObject* value, void*){
BOB_TRY
  self->cxx->use_DFFS(PyObject_IsTrue(value));
  return 0;
BOB_CATCH_MEMBER("use_DFFS", -1)
}

static auto input_size_doc = bob::extension::VariableDoc(
  "input_size",
  "int",
  "The expected input dimensionality, read-only"
);
PyObject* PyBobLearnLinearBICMachine_getInputSize(PyBobLearnLinearBICMachineObject* self, void*){
BOB_TRY
  return Py_BuildValue("i", self->cxx->input_size());
BOB_CATCH_MEMBER("input_size", 0)
}


static PyGetSetDef PyBobLearnLinearBICMachine_getseters[] = {
  {
    dffs_doc.name(),
    (getter)PyBobLearnLinearBICMachine_getDFFS,
    (setter)PyBobLearnLinearBICMachine_setDFFS,
    dffs_doc.doc(),
    0
  },
  {
    input_size_doc.name(),
    (getter)PyBobLearnLinearBICMachine_getInputSize,
    0,
    input_size_doc.doc(),
    0
  },
  {0}  /* Sentinel */
};

/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto forward_doc = bob::extension::FunctionDoc(
  "forward",
  "Computes the BIC or IEC score for the given input vector, which results of a comparison vector of two (facial) images",
  "The resulting value is returned as a single float value. "
  "The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class.\n\n"
  ".. note:: the ``__call__`` method is an alias for this one",
  true
)
.add_prototype("input", "score")
.add_parameter("input", "array_like (float, 1D)", "The input vector, which is the result of comparing to (facial) images")
.add_return("score", "float", "The log-likelihood that the given ``input`` belongs to the intrapersonal class")
;

static PyObject* PyBobLearnLinearBICMachine_forward(PyBobLearnLinearBICMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = forward_doc.kwlist();

  PyBlitzArrayObject* input;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBlitzArray_Converter, &input)) return 0;
  auto input_ = make_safe(input);

  if (input->ndim != 1 || input->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 1D 64-bit float arrays for 'input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  double score = self->cxx->forward(*PyBlitzArrayCxx_AsBlitz<double,1>(input));
  return Py_BuildValue("d", score);
BOB_CATCH_MEMBER("forward", 0)
}

static auto similar_doc = bob::extension::FunctionDoc(
  "is_similar_to",
  "Compares this BICMachine with the ``other`` one to be approximately the same",
  "The optional values ``r_epsilon`` and ``a_epsilon`` refer to the relative and absolute precision, similarly to :py:func:`numpy.allclose`.",
  true
)
.add_prototype("other, [r_epsilon], [a_epsilon]", "similar")
.add_parameter("other", ":py:class:`bob.learn.linear.BICMachine`", "The other BICMachine to compare with")
.add_parameter("r_epsilon", "float", "[Default: ``1e-5``] The relative precision")
.add_parameter("a_epsilon", "float", "[Default: ``1e-8``] The absolute precision")
.add_return("similar", "bool", "``True`` if the ``other`` machine is similar to this one, otherwise ``False``")
;
static PyObject* PyBobLearnLinearBICMachine_similar(PyBobLearnLinearBICMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = similar_doc.kwlist();

  PyBobLearnLinearBICMachineObject* other = 0;
  double r_epsilon = 1.e-5;
  double a_epsilon = 1.e-8;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|dd", kwlist, &PyBobLearnLinearBICMachine_Type, &other, &r_epsilon, &a_epsilon)) return 0;

  if (self->cxx->is_similar_to(*other->cxx, r_epsilon, a_epsilon))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
BOB_CATCH_MEMBER("is_similar_to", 0)
}

static auto load_doc = bob::extension::FunctionDoc(
  "load",
  "Loads the BIC machine from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file opened for reading")
;

static PyObject* PyBobLearnLinearBICMachine_load(PyBobLearnLinearBICMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = load_doc.kwlist();
  PyBobIoHDF5FileObject* file;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, PyBobIoHDF5File_Converter, &file)) return 0;

  auto file_ = make_safe(file);
  self->cxx->load(*file->f);
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("load", 0)
}


static auto save_doc = bob::extension::FunctionDoc(
  "save",
  "Saves the BIC machine to the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "An HDF5 file open for writing")
;

static PyObject* PyBobLearnLinearBICMachine_save(PyBobLearnLinearBICMachineObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = save_doc.kwlist();
  PyBobIoHDF5FileObject* file;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,"O&", kwlist, PyBobIoHDF5File_Converter, &file)) return 0;

  auto file_ = make_safe(file);
  self->cxx->save(*file->f);
  Py_RETURN_NONE;
BOB_CATCH_MEMBER("save", 0)
}



static PyMethodDef PyBobLearnLinearBICMachine_methods[] = {
  {
    forward_doc.name(),
    (PyCFunction)PyBobLearnLinearBICMachine_forward,
    METH_VARARGS|METH_KEYWORDS,
    forward_doc.doc()
  },
  {
    similar_doc.name(),
    (PyCFunction)PyBobLearnLinearBICMachine_similar,
    METH_VARARGS|METH_KEYWORDS,
    similar_doc.doc()
  },
  {
    load_doc.name(),
    (PyCFunction)PyBobLearnLinearBICMachine_load,
    METH_VARARGS|METH_KEYWORDS,
    load_doc.doc()
  },
  {
    save_doc.name(),
    (PyCFunction)PyBobLearnLinearBICMachine_save,
    METH_VARARGS|METH_KEYWORDS,
    save_doc.doc()
  },
  {0} /* Sentinel */
};


static auto train_doc = bob::extension::FunctionDoc(
  "train",
  "Trains the given machine to classify intrapersonal (image) difference vectors vs. extrapersonal ones",
  "The given difference vectors might be the result of any (image) comparison function, e.g., the pixel difference of two images. "
  "In any case, all distance vectors must have the same length.",
  true
)
.add_prototype("intra_differences, extra_differences, [machine]", "machine")
.add_parameter("intra_differences", "array_like (float, 2D)", "The input vectors, which are the result of intrapersonal (facial image) comparisons, in shape ``(#features, length)``")
.add_parameter("extra_differences", "array_like (float, 2D)", "The input vectors, which are the result of extrapersonal (facial image) comparisons, in shape ``(#features, length)``")
.add_parameter("machine", ":py:class:`bob.learn.linear.BICMachine`", "The machine to be trained")
.add_return("machine", ":py:class:`bob.learn.linear.BICMachine`", "A newly generated and trained BIC machine, where the `bob.lear.linear.BICMachine.use_DFFS` flag is set to ``False``")
;

static PyObject* PyBobLearnLinearBICTrainer_train(PyBobLearnLinearBICTrainerObject* self, PyObject* args, PyObject* kwargs) {
BOB_TRY
  char** kwlist = train_doc.kwlist();

  PyBlitzArrayObject* intra,* extra;
  PyBobLearnLinearBICMachineObject* machine = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|O!", kwlist, &PyBlitzArray_Converter, &intra, &PyBlitzArray_Converter, &extra, &PyBobLearnLinearBICMachine_Type, &machine)) return 0;

  auto intra_ = make_safe(intra), extra_ = make_safe(extra);
  boost::shared_ptr<PyBobLearnLinearBICMachineObject> machine_;

  if (intra->ndim != 2 || intra->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for 'intra_differences'", Py_TYPE(self)->tp_name);
    return 0;
  }
  if (extra->ndim != 2 || extra->type_num != NPY_FLOAT64){
    PyErr_Format(PyExc_TypeError, "`%s' only supports 2D 64-bit float arrays for 'extra_differences'", Py_TYPE(self)->tp_name);
    return 0;
  }
  if (intra->shape[1] != extra->shape[1]){
    PyErr_Format(PyExc_TypeError, "`%s' The lenght of the feature vectors differ", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (!machine){
    // create machine if not given
    machine = (PyBobLearnLinearBICMachineObject*)PyBobLearnLinearBICMachine_Type.tp_alloc(&PyBobLearnLinearBICMachine_Type, 0);
    machine_ = make_safe(machine);
    machine->cxx.reset(new bob::learn::linear::BICMachine());
  }

  // train it
  self->cxx->train(*machine->cxx, *PyBlitzArrayCxx_AsBlitz<double,2>(intra), *PyBlitzArrayCxx_AsBlitz<double,2>(extra));
  return Py_BuildValue("O", machine);
BOB_CATCH_MEMBER("train", 0)
}

static PyMethodDef PyBobLearnLinearBICTrainer_methods[] = {
  {
    train_doc.name(),
    (PyCFunction)PyBobLearnLinearBICTrainer_train,
    METH_VARARGS|METH_KEYWORDS,
    train_doc.doc()
  },
  {0} /* Sentinel */
};

/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// BIC Machine
PyTypeObject PyBobLearnLinearBICMachine_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

// BIC Trainer
PyTypeObject PyBobLearnLinearBICTrainer_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};


bool init_BobLearnLinearBIC(PyObject* module)
{

  // BIC Machine
  PyBobLearnLinearBICMachine_Type.tp_name = BICMachine_doc.name();
  PyBobLearnLinearBICMachine_Type.tp_basicsize = sizeof(PyBobLearnLinearBICMachineObject);
  PyBobLearnLinearBICMachine_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearBICMachine_Type.tp_doc = BICMachine_doc.doc();

  // set the functions
  PyBobLearnLinearBICMachine_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearBICMachine_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearBICMachine_init);
  PyBobLearnLinearBICMachine_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearBICMachine_delete);
  PyBobLearnLinearBICMachine_Type.tp_methods = PyBobLearnLinearBICMachine_methods;
  PyBobLearnLinearBICMachine_Type.tp_getset = PyBobLearnLinearBICMachine_getseters;
  PyBobLearnLinearBICMachine_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobLearnLinearBICMachine_forward);
  PyBobLearnLinearBICMachine_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobLearnLinearBICMachine_RichCompare);

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearBICMachine_Type) < 0)
    return false;

  // BIC Trainer
  PyBobLearnLinearBICTrainer_Type.tp_name = BICTrainer_doc.name();
  PyBobLearnLinearBICTrainer_Type.tp_basicsize = sizeof(PyBobLearnLinearBICTrainerObject);
  PyBobLearnLinearBICTrainer_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobLearnLinearBICTrainer_Type.tp_doc = BICTrainer_doc.doc();

  // set the functions
  PyBobLearnLinearBICTrainer_Type.tp_new = PyType_GenericNew;
  PyBobLearnLinearBICTrainer_Type.tp_init = reinterpret_cast<initproc>(PyBobLearnLinearBICTrainer_init);
  PyBobLearnLinearBICTrainer_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobLearnLinearBICTrainer_delete);
  PyBobLearnLinearBICTrainer_Type.tp_methods = PyBobLearnLinearBICTrainer_methods;

  // check that everyting is fine
  if (PyType_Ready(&PyBobLearnLinearBICTrainer_Type) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobLearnLinearBICMachine_Type);
  Py_INCREF(&PyBobLearnLinearBICTrainer_Type);
  return
    PyModule_AddObject(module, "BICMachine", (PyObject*)&PyBobLearnLinearBICMachine_Type) >= 0 &&
    PyModule_AddObject(module, "BICTrainer", (PyObject*)&PyBobLearnLinearBICTrainer_Type) >= 0;
}
