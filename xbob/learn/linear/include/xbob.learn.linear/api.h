/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief C/C++ API for bob::machine
 */

#ifndef XBOB_LEARN_LINEAR_H
#define XBOB_LEARN_LINEAR_H

#include <Python.h>
#include <xbob.learn.linear/config.h>
#include <xbob.learn.linear/machine.h>
#include <xbob.learn.linear/pca.h>
#include <xbob.learn.linear/lda.h>
#include <xbob.learn.linear/logreg.h>
#include <xbob.learn.linear/whitening.h>

#define XBOB_LEARN_LINEAR_MODULE_PREFIX xbob.learn.linear
#define XBOB_LEARN_LINEAR_MODULE_NAME _library

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobLearnLinear_ENUM{
  PyXbobLearnLinear_APIVersion_NUM = 0,
  // Bindings for xbob.learn.linear.Machine
  PyBobLearnLinearMachine_Type_NUM,
  PyBobLearnLinearMachine_Check_NUM,
  PyBobLearnLinearMachine_NewFromSize_NUM,
  // Bindings for xbob.learn.linear.PCATrainer
  PyBobLearnLinearPCATrainer_Type_NUM,
  PyBobLearnLinearPCATrainer_Check_NUM,
  // Bindings for xbob.learn.linear.FisherLDATrainer
  PyBobLearnLinearFisherLDATrainer_Type_NUM,
  PyBobLearnLinearFisherLDATrainer_Check_NUM,
  // Bindings for xbob.learn.linear.CGLogRegTrainer
  PyBobLearnLinearCGLogRegTrainer_Type_NUM,
  PyBobLearnLinearCGLogRegTrainer_Check_NUM,
  // Bindings for xbob.learn.linear.WhiteningTrainer
  PyBobLearnLinearWhiteningTrainer_Type_NUM,
  PyBobLearnLinearWhiteningTrainer_Check_NUM,
  // Total number of C API pointers
  PyXbobLearnLinear_API_pointers
};

/**************
 * Versioning *
 **************/

#define PyXbobLearnLinear_APIVersion_TYPE int

/******************************************
 * Bindings for xbob.learn.linear.Machine *
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
 * Bindings for xbob.learn.linear.PCATrainer *
 *********************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::PCATrainer* cxx;
} PyBobLearnLinearPCATrainerObject;

#define PyBobLearnLinearPCATrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearPCATrainer_Check_RET int
#define PyBobLearnLinearPCATrainer_Check_PROTO (PyObject* o)

/***************************************************
 * Bindings for xbob.learn.linear.FisherLDATrainer *
 ***************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::FisherLDATrainer* cxx;
} PyBobLearnLinearFisherLDATrainerObject;

#define PyBobLearnLinearFisherLDATrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearFisherLDATrainer_Check_RET int
#define PyBobLearnLinearFisherLDATrainer_Check_PROTO (PyObject* o)

/**************************************************
 * Bindings for xbob.learn.linear.CGLogRegTrainer *
 **************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::CGLogRegTrainer* cxx;
} PyBobLearnLinearCGLogRegTrainerObject;

#define PyBobLearnLinearCGLogRegTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearCGLogRegTrainer_Check_RET int
#define PyBobLearnLinearCGLogRegTrainer_Check_PROTO (PyObject* o)

/***************************************************
 * Bindings for xbob.learn.linear.WhiteningTrainer *
 ***************************************************/

typedef struct {
  PyObject_HEAD
  bob::learn::linear::WhiteningTrainer* cxx;
} PyBobLearnLinearWhiteningTrainerObject;

#define PyBobLearnLinearWhiteningTrainer_Type_TYPE PyTypeObject

#define PyBobLearnLinearWhiteningTrainer_Check_RET int
#define PyBobLearnLinearWhiteningTrainer_Check_PROTO (PyObject* o)


#ifdef XBOB_LEARN_LINEAR_MODULE

  /* This section is used when compiling `xbob.learn.linear' itself */

  /**************
   * Versioning *
   **************/

  extern int PyXbobLearnLinear_APIVersion;

  /******************************************
   * Bindings for xbob.learn.linear.Machine *
   ******************************************/

  extern PyBobLearnLinearMachine_Type_TYPE PyBobLearnLinearMachine_Type;

  PyBobLearnLinearMachine_Check_RET PyBobLearnLinearMachine_Check PyBobLearnLinearMachine_Check_PROTO;

  PyBobLearnLinearMachine_NewFromSize_RET PyBobLearnLinearMachine_NewFromSize PyBobLearnLinearMachine_NewFromSize_PROTO;

  /*********************************************
   * Bindings for xbob.learn.linear.PCATrainer *
   *********************************************/

  extern PyBobLearnLinearPCATrainer_Type_TYPE PyBobLearnLinearPCATrainer_Type;

  PyBobLearnLinearPCATrainer_Check_RET PyBobLearnLinearPCATrainer_Check PyBobLearnLinearPCATrainer_Check_PROTO;

  /***************************************************
   * Bindings for xbob.learn.linear.FisherLDATrainer *
   ***************************************************/

  extern PyBobLearnLinearFisherLDATrainer_Type_TYPE PyBobLearnLinearFisherLDATrainer_Type;

  PyBobLearnLinearFisherLDATrainer_Check_RET PyBobLearnLinearFisherLDATrainer_Check PyBobLearnLinearFisherLDATrainer_Check_PROTO;

  /**************************************************
   * Bindings for xbob.learn.linear.CGLogRegTrainer *
   **************************************************/

  extern PyBobLearnLinearCGLogRegTrainer_Type_TYPE PyBobLearnLinearCGLogRegTrainer_Type;

  PyBobLearnLinearCGLogRegTrainer_Check_RET PyBobLearnLinearCGLogRegTrainer_Check PyBobLearnLinearCGLogRegTrainer_Check_PROTO;

  /***************************************************
   * Bindings for xbob.learn.linear.WhiteningTrainer *
   ***************************************************/

  extern PyBobLearnLinearWhiteningTrainer_Type_TYPE PyBobLearnLinearWhiteningTrainer_Type;

  PyBobLearnLinearWhiteningTrainer_Check_RET PyBobLearnLinearWhiteningTrainer_Check PyBobLearnLinearWhiteningTrainer_Check_PROTO;

#else

  /* This section is used in modules that use `xbob.learn.linear's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define XBOB_LEARN_LINEAR_MAKE_API_NAME_INNER(a) XBOB_LEARN_LINEAR_ ## a
#    define XBOB_LEARN_LINEAR_MAKE_API_NAME(a) XBOB_LEARN_LINEAR_MAKE_API_NAME_INNER(a)
#    define PyXbobLearnLinear_API XBOB_LEARN_LINEAR_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobLearnLinear_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobLearnLinear_API;
#    else
  static void **PyXbobLearnLinear_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyXbobLearnLinear_APIVersion (*(PyXbobLearnLinear_APIVersion_TYPE *)PyXbobLearnLinear_API[PyXbobLearnLinear_APIVersion_NUM])

  /******************************************
   * Bindings for xbob.learn.linear.Machine *
   ******************************************/

# define PyBobLearnLinearMachine_Type (*(PyBobLearnLinearMachine_Type_TYPE *)PyXbobLearnLinear_API[PyBobLearnLinearMachine_Type_NUM])

# define PyBobLearnLinearMachine_Check (*(PyBobLearnLinearMachine_Check_RET (*)PyBobLearnLinearMachine_Check_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearMachine_Check_NUM])

# define PyBobLearnLinearMachine_NewFromSize (*(PyBobLearnLinearMachine_NewFromSize_RET (*)PyBobLearnLinearMachine_NewFromSize_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearMachine_NewFromSize_NUM])

  /*********************************************
   * Bindings for xbob.learn.linear.PCATrainer *
   *********************************************/

# define PyBobLearnLinearPCATrainer_Type (*(PyBobLearnLinearPCATrainer_Type_TYPE *)PyXbobLearnLinear_API[PyBobLearnLinearPCATrainer_Type_NUM])

# define PyBobLearnLinearPCATrainer_Check (*(PyBobLearnLinearPCATrainer_Check_RET (*)PyBobLearnLinearPCATrainer_Check_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearPCATrainer_Check_NUM])

  /***************************************************
   * Bindings for xbob.learn.linear.FisherLDATrainer *
   ***************************************************/

# define PyBobLearnLinearFisherLDATrainer_Type (*(PyBobLearnLinearFisherLDATrainer_Type_TYPE *)PyXbobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Type_NUM])

# define PyBobLearnLinearFisherLDATrainer_Check (*(PyBobLearnLinearFisherLDATrainer_Check_RET (*)PyBobLearnLinearFisherLDATrainer_Check_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Check_NUM])

  /**************************************************
   * Bindings for xbob.learn.linear.CGLogRegTrainer *
   **************************************************/

# define PyBobLearnLinearCGLogRegTrainer_Type (*(PyBobLearnLinearCGLogRegTrainer_Type_TYPE *)PyXbobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Type_NUM])

# define PyBobLearnLinearCGLogRegTrainer_Check (*(PyBobLearnLinearCGLogRegTrainer_Check_RET (*)PyBobLearnLinearCGLogRegTrainer_Check_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Check_NUM])

  /***************************************************
   * Bindings for xbob.learn.linear.WhiteningTrainer *
   ***************************************************/

# define PyBobLearnLinearWhiteningTrainer_Type (*(PyBobLearnLinearWhiteningTrainer_Type_TYPE *)PyXbobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Type_NUM])

# define PyBobLearnLinearWhiteningTrainer_Check (*(PyBobLearnLinearWhiteningTrainer_Check_RET (*)PyBobLearnLinearWhiteningTrainer_Check_PROTO) PyXbobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Check_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_xbob_learn_linear(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_NAME));

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyXbobLearnLinear_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      XbobLearnLinear_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!XbobLearnLinear_API) {
      PyErr_Format(PyExc_ImportError,
#   if PY_VERSION_HEX >= 0x02070000
          "cannot find C/C++ API capsule at `%s.%s._C_API'",
#   else
          "cannot find C/C++ API cobject at `%s.%s._C_API'",
#   endif
          BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_PREFIX),
          BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_NAME));
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobLearnLinear_API[PyXbobLearnLinear_APIVersion_NUM];

    if (XBOB_LEARN_LINEAR_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, "%s.%s import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_PREFIX), BOOST_PP_STRINGIZE(XBOB_LEARN_LINEAR_MODULE_NAME), XBOB_LEARN_LINEAR_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* XBOB_LEARN_LINEAR_MODULE */

#endif /* XBOB_LEARN_LINEAR_H */
