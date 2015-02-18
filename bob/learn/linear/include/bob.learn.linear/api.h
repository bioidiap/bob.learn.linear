/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief C/C++ API for bob::machine
 */

#ifndef BOB_LEARN_LINEAR_H
#define BOB_LEARN_LINEAR_H

#include <Python.h>
#include <bob.learn.linear/config.h>
#include <bob.learn.linear/machine.h>
#include <bob.learn.linear/pca.h>
#include <bob.learn.linear/lda.h>
#include <bob.learn.linear/logreg.h>
#include <bob.learn.linear/whitening.h>
#include <bob.learn.linear/wccn.h>
#include <bob.learn.linear/bic.h>

#define BOB_LEARN_LINEAR_MODULE_PREFIX bob.learn.linear
#define BOB_LEARN_LINEAR_MODULE_NAME _library

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobLearnLinear_ENUM{
  PyBobLearnLinear_APIVersion_NUM = 0,
  // Bindings for bob.learn.linear.Machine
  PyBobLearnLinearMachine_Type_NUM,
  PyBobLearnLinearMachine_Check_NUM,
  PyBobLearnLinearMachine_NewFromSize_NUM,
  // Bindings for bob.learn.linear.PCATrainer
  PyBobLearnLinearPCATrainer_Type_NUM,
  PyBobLearnLinearPCATrainer_Check_NUM,
  // Bindings for bob.learn.linear.FisherLDATrainer
  PyBobLearnLinearFisherLDATrainer_Type_NUM,
  PyBobLearnLinearFisherLDATrainer_Check_NUM,
  // Bindings for bob.learn.linear.CGLogRegTrainer
  PyBobLearnLinearCGLogRegTrainer_Type_NUM,
  PyBobLearnLinearCGLogRegTrainer_Check_NUM,
  // Bindings for bob.learn.linear.WhiteningTrainer
  PyBobLearnLinearWhiteningTrainer_Type_NUM,
  PyBobLearnLinearWhiteningTrainer_Check_NUM,
  // Bindings for bob.learn.linear.WCCNTrainer
  PyBobLearnLinearWCCNTrainer_Type_NUM,
  PyBobLearnLinearWCCNTrainer_Check_NUM,
  // Bindings for bob.learn.linear.BICMachine
  PyBobLearnLinearBICMachine_Type_NUM,
  PyBobLearnLinearBICMachine_Check_NUM,
  // Bindings for bob.learn.linear.BICTrainer
  PyBobLearnLinearBICTrainer_Type_NUM,
  PyBobLearnLinearBICTrainer_Check_NUM,
  // Total number of C API pointers
  PyBobLearnLinear_API_pointers
};

/**************
 * Versioning *
 **************/

#define PyBobLearnLinear_APIVersion_TYPE int

/******************************************
 * Bindings for bob.learn.linear.Machine *
 ******************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::Machine* cxx;
} PyBobLearnLinearMachineObject;

#define PyBobLearnLinearMachine_Type_TYPE PyTypeObject

#define PyBobLearnLinearMachine_Check_RET int
#define PyBobLearnLinearMachine_Check_PROTO (PyObject* o)

#define PyBobLearnLinearMachine_NewFromSize_RET PyObject*
#define PyBobLearnLinearMachine_NewFromSize_PROTO (Py_ssize_t i, Py_ssize_t o)

/*********************************************
 * Bindings for bob.learn.linear.PCATrainer *
 *********************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::PCATrainer* cxx;
} PyBobLearnLinearPCATrainerObject;

#define PyBobLearnLinearPCATrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearPCATrainer_Check_RET int
#define PyBobLearnLinearPCATrainer_Check_PROTO (PyObject* o)

/***************************************************
 * Bindings for bob.learn.linear.FisherLDATrainer *
 ***************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::FisherLDATrainer* cxx;
} PyBobLearnLinearFisherLDATrainerObject;

#define PyBobLearnLinearFisherLDATrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearFisherLDATrainer_Check_RET int
#define PyBobLearnLinearFisherLDATrainer_Check_PROTO (PyObject* o)

/**************************************************
 * Bindings for bob.learn.linear.CGLogRegTrainer *
 **************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::CGLogRegTrainer* cxx;
} PyBobLearnLinearCGLogRegTrainerObject;

#define PyBobLearnLinearCGLogRegTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearCGLogRegTrainer_Check_RET int
#define PyBobLearnLinearCGLogRegTrainer_Check_PROTO (PyObject* o)

/***************************************************
 * Bindings for bob.learn.linear.WhiteningTrainer *
 ***************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::WhiteningTrainer* cxx;
} PyBobLearnLinearWhiteningTrainerObject;

#define PyBobLearnLinearWhiteningTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearWhiteningTrainer_Check_RET int
#define PyBobLearnLinearWhiteningTrainer_Check_PROTO (PyObject* o)

/**********************************************
 * Bindings for bob.learn.linear.WCCNTrainer *
 **********************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::WCCNTrainer* cxx;
} PyBobLearnLinearWCCNTrainerObject;

#define PyBobLearnLinearWCCNTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearWCCNTrainer_Check_RET int
#define PyBobLearnLinearWCCNTrainer_Check_PROTO (PyObject* o)


/**********************************************
 * Bindings for bob.learn.linear.BICMachine *
 **********************************************/

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::linear::BICMachine> cxx;
} PyBobLearnLinearBICMachineObject;

#define PyBobLearnLinearBICMachine_Type_TYPE PyTypeObject

#define PyBobLearnLinearBICMachine_Check_RET int
#define PyBobLearnLinearBICMachine_Check_PROTO (PyObject* o)


/**********************************************
 * Bindings for bob.learn.linear.BICTrainer *
 **********************************************/

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::linear::BICTrainer> cxx;
} PyBobLearnLinearBICTrainerObject;

#define PyBobLearnLinearBICTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearBICTrainer_Check_RET int
#define PyBobLearnLinearBICTrainer_Check_PROTO (PyObject* o)


#ifdef BOB_LEARN_LINEAR_MODULE

  /* This section is used when compiling `bob.learn.linear' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobLearnLinear_APIVersion;

  /******************************************
   * Bindings for bob.learn.linear.Machine *
   ******************************************/

  extern PyBobLearnLinearMachine_Type_TYPE PyBobLearnLinearMachine_Type;

  PyBobLearnLinearMachine_Check_RET PyBobLearnLinearMachine_Check PyBobLearnLinearMachine_Check_PROTO;

  PyBobLearnLinearMachine_NewFromSize_RET PyBobLearnLinearMachine_NewFromSize PyBobLearnLinearMachine_NewFromSize_PROTO;

  /*********************************************
   * Bindings for bob.learn.linear.PCATrainer *
   *********************************************/

  extern PyBobLearnLinearPCATrainer_Type_TYPE PyBobLearnLinearPCATrainer_Type;

  PyBobLearnLinearPCATrainer_Check_RET PyBobLearnLinearPCATrainer_Check PyBobLearnLinearPCATrainer_Check_PROTO;

  /***************************************************
   * Bindings for bob.learn.linear.FisherLDATrainer *
   ***************************************************/

  extern PyBobLearnLinearFisherLDATrainer_Type_TYPE PyBobLearnLinearFisherLDATrainer_Type;

  PyBobLearnLinearFisherLDATrainer_Check_RET PyBobLearnLinearFisherLDATrainer_Check PyBobLearnLinearFisherLDATrainer_Check_PROTO;

  /**************************************************
   * Bindings for bob.learn.linear.CGLogRegTrainer *
   **************************************************/

  extern PyBobLearnLinearCGLogRegTrainer_Type_TYPE PyBobLearnLinearCGLogRegTrainer_Type;

  PyBobLearnLinearCGLogRegTrainer_Check_RET PyBobLearnLinearCGLogRegTrainer_Check PyBobLearnLinearCGLogRegTrainer_Check_PROTO;

  /***************************************************
   * Bindings for bob.learn.linear.WhiteningTrainer *
   ***************************************************/

  extern PyBobLearnLinearWhiteningTrainer_Type_TYPE PyBobLearnLinearWhiteningTrainer_Type;

  PyBobLearnLinearWhiteningTrainer_Check_RET PyBobLearnLinearWhiteningTrainer_Check PyBobLearnLinearWhiteningTrainer_Check_PROTO;

  /**********************************************
   * Bindings for bob.learn.linear.WCCNTrainer *
   **********************************************/

  extern PyBobLearnLinearWCCNTrainer_Type_TYPE PyBobLearnLinearWCCNTrainer_Type;

  PyBobLearnLinearWCCNTrainer_Check_RET PyBobLearnLinearWCCNTrainer_Check PyBobLearnLinearWCCNTrainer_Check_PROTO;

  /**********************************************
   * Bindings for bob.learn.linear.BICMachine *
   **********************************************/

  extern PyBobLearnLinearBICMachine_Type_TYPE PyBobLearnLinearBICMachine_Type;

  PyBobLearnLinearBICMachine_Check_RET PyBobLearnLinearBICMachine_Check PyBobLearnLinearBICMachine_Check_PROTO;

  /**********************************************
   * Bindings for bob.learn.linear.BICTrainer *
   **********************************************/

  extern PyBobLearnLinearBICTrainer_Type_TYPE PyBobLearnLinearBICTrainer_Type;

  PyBobLearnLinearBICTrainer_Check_RET PyBobLearnLinearBICTrainer_Check PyBobLearnLinearBICTrainer_Check_PROTO;

#else

  /* This section is used in modules that use `bob.learn.linear's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define BOB_LEARN_LINEAR_MAKE_API_NAME_INNER(a) BOB_LEARN_LINEAR_ ## a
#    define BOB_LEARN_LINEAR_MAKE_API_NAME(a) BOB_LEARN_LINEAR_MAKE_API_NAME_INNER(a)
#    define PyBobLearnLinear_API BOB_LEARN_LINEAR_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyBobLearnLinear_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobLearnLinear_API;
#    else
  static void **PyBobLearnLinear_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyBobLearnLinear_APIVersion (*(PyBobLearnLinear_APIVersion_TYPE *)PyBobLearnLinear_API[PyBobLearnLinear_APIVersion_NUM])

  /******************************************
   * Bindings for bob.learn.linear.Machine *
   ******************************************/

# define PyBobLearnLinearMachine_Type (*(PyBobLearnLinearMachine_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearMachine_Type_NUM])

# define PyBobLearnLinearMachine_Check (*(PyBobLearnLinearMachine_Check_RET (*)PyBobLearnLinearMachine_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearMachine_Check_NUM])

# define PyBobLearnLinearMachine_NewFromSize (*(PyBobLearnLinearMachine_NewFromSize_RET (*)PyBobLearnLinearMachine_NewFromSize_PROTO) PyBobLearnLinear_API[PyBobLearnLinearMachine_NewFromSize_NUM])

  /*********************************************
   * Bindings for bob.learn.linear.PCATrainer *
   *********************************************/

# define PyBobLearnLinearPCATrainer_Type (*(PyBobLearnLinearPCATrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearPCATrainer_Type_NUM])

# define PyBobLearnLinearPCATrainer_Check (*(PyBobLearnLinearPCATrainer_Check_RET (*)PyBobLearnLinearPCATrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearPCATrainer_Check_NUM])

  /***************************************************
   * Bindings for bob.learn.linear.FisherLDATrainer *
   ***************************************************/

# define PyBobLearnLinearFisherLDATrainer_Type (*(PyBobLearnLinearFisherLDATrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Type_NUM])

# define PyBobLearnLinearFisherLDATrainer_Check (*(PyBobLearnLinearFisherLDATrainer_Check_RET (*)PyBobLearnLinearFisherLDATrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Check_NUM])

  /**************************************************
   * Bindings for bob.learn.linear.CGLogRegTrainer *
   **************************************************/

# define PyBobLearnLinearCGLogRegTrainer_Type (*(PyBobLearnLinearCGLogRegTrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Type_NUM])

# define PyBobLearnLinearCGLogRegTrainer_Check (*(PyBobLearnLinearCGLogRegTrainer_Check_RET (*)PyBobLearnLinearCGLogRegTrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Check_NUM])

  /***************************************************
   * Bindings for bob.learn.linear.WhiteningTrainer *
   ***************************************************/

# define PyBobLearnLinearWhiteningTrainer_Type (*(PyBobLearnLinearWhiteningTrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Type_NUM])

# define PyBobLearnLinearWhiteningTrainer_Check (*(PyBobLearnLinearWhiteningTrainer_Check_RET (*)PyBobLearnLinearWhiteningTrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Check_NUM])

  /**********************************************
   * Bindings for bob.learn.linear.WCCNTrainer *
   **********************************************/

# define PyBobLearnLinearWCCNTrainer_Type (*(PyBobLearnLinearWCCNTrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearWCCNTrainer_Type_NUM])

# define PyBobLearnLinearWCCNTrainer_Check (*(PyBobLearnLinearWCCNTrainer_Check_RET (*)PyBobLearnLinearWCCNTrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearWCCNTrainer_Check_NUM])

  /**********************************************
   * Bindings for bob.learn.linear.BICMachine *
   **********************************************/

# define PyBobLearnLinearBICMachine_Type (*(PyBobLearnLinearBICMachine_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearBICMachine_Type_NUM])

# define PyBobLearnLinearBICMachine_Check (*(PyBobLearnLinearBICMachine_Check_RET (*)PyBobLearnLinearBICMachine_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearBICMachine_Check_NUM])

  /**********************************************
   * Bindings for bob.learn.linear.BICTrainer *
   **********************************************/

# define PyBobLearnLinearBICTrainer_Type (*(PyBobLearnLinearBICTrainer_Type_TYPE *)PyBobLearnLinear_API[PyBobLearnLinearBICTrainer_Type_NUM])

# define PyBobLearnLinearBICTrainer_Check (*(PyBobLearnLinearBICTrainer_Check_RET (*)PyBobLearnLinearBICTrainer_Check_PROTO) PyBobLearnLinear_API[PyBobLearnLinearBICTrainer_Check_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_bob_learn_linear(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_NAME));

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobLearnLinear_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyBobLearnLinear_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobLearnLinear_API) {
      PyErr_Format(PyExc_ImportError,
#   if PY_VERSION_HEX >= 0x02070000
          "cannot find C/C++ API capsule at `%s.%s._C_API'",
#   else
          "cannot find C/C++ API cobject at `%s.%s._C_API'",
#   endif
          BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_PREFIX),
          BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_NAME));
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobLearnLinear_API[PyBobLearnLinear_APIVersion_NUM];

    if (BOB_LEARN_LINEAR_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, "%s.%s import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_PREFIX), BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_MODULE_NAME), BOB_LEARN_LINEAR_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_LEARN_LINEAR_MODULE */

#endif /* BOB_LEARN_LINEAR_H */
