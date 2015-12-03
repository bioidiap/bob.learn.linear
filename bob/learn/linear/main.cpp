/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 13 Dec 2013 12:35:59 CET
 *
 * @brief Bindings to bob::learn::linear
 */

#define BOB_LEARN_LINEAR_MODULE
#include <bob.learn.linear/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.learn.activation/api.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Bob's Linear machine and trainers");

int PyBobLearnLinear_APIVersion = BOB_LEARN_LINEAR_API_VERSION;

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

extern bool init_BobLearnLinearMachine(PyObject* module);
extern bool init_BobLearnLinearPCA(PyObject* module);
extern bool init_BobLearnLinearLDA(PyObject* module);
extern bool init_BobLearnLinearCGLogReg(PyObject* module);
extern bool init_BobLearnLinearWCCN(PyObject* module);
extern bool init_BobLearnLinearWhitening(PyObject* module);
extern bool init_BobLearnLinearBIC(PyObject* module);

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  /* register the types to python */
  if (!init_BobLearnLinearMachine(module)) return 0;
  if (!init_BobLearnLinearPCA(module)) return 0;
  if (!init_BobLearnLinearLDA(module)) return 0;
  if (!init_BobLearnLinearCGLogReg(module)) return 0;
  if (!init_BobLearnLinearWCCN(module)) return 0;
  if (!init_BobLearnLinearWhitening(module)) return 0;
  if (!init_BobLearnLinearBIC(module)) return 0;
  static void* PyBobLearnLinear_API[PyBobLearnLinear_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyBobLearnLinear_API[PyBobLearnLinear_APIVersion_NUM] = (void *)&PyBobLearnLinear_APIVersion;

  /******************************************
   * Bindings for bob.learn.linear.Machine *
   ******************************************/

  PyBobLearnLinear_API[PyBobLearnLinearMachine_Type_NUM] = (void *)&PyBobLearnLinearMachine_Type;

  PyBobLearnLinear_API[PyBobLearnLinearMachine_Check_NUM] = (void *)&PyBobLearnLinearMachine_Check;

  PyBobLearnLinear_API[PyBobLearnLinearMachine_NewFromSize_NUM] = (void *)&PyBobLearnLinearMachine_NewFromSize;

  /*********************************************
   * Bindings for bob.learn.linear.PCATrainer *
   *********************************************/

  PyBobLearnLinear_API[PyBobLearnLinearPCATrainer_Type_NUM] = (void *)&PyBobLearnLinearPCATrainer_Type;

  PyBobLearnLinear_API[PyBobLearnLinearPCATrainer_Check_NUM] = (void *)&PyBobLearnLinearPCATrainer_Check;

  /***************************************************
   * Bindings for bob.learn.linear.FisherLDATrainer *
   ***************************************************/

  PyBobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Type_NUM] = (void *)&PyBobLearnLinearFisherLDATrainer_Type;

  PyBobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Check_NUM] = (void *)&PyBobLearnLinearFisherLDATrainer_Check;

  /**************************************************
   * Bindings for bob.learn.linear.CGLogRegTrainer *
   **************************************************/

  PyBobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Type_NUM] = (void *)&PyBobLearnLinearCGLogRegTrainer_Type;

  PyBobLearnLinear_API[PyBobLearnLinearCGLogRegTrainer_Check_NUM] = (void *)&PyBobLearnLinearCGLogRegTrainer_Check;

  /***************************************************
   * Bindings for bob.learn.linear.WhiteningTrainer *
   ***************************************************/

  PyBobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Type_NUM] = (void *)&PyBobLearnLinearWhiteningTrainer_Type;

  PyBobLearnLinear_API[PyBobLearnLinearWhiteningTrainer_Check_NUM] = (void *)&PyBobLearnLinearWhiteningTrainer_Check;

  /***************************************************
   * Bindings for bob.learn.linear.WCCNTrainer *
   ***************************************************/

  PyBobLearnLinear_API[PyBobLearnLinearWCCNTrainer_Type_NUM] = (void *)&PyBobLearnLinearWCCNTrainer_Type;

  PyBobLearnLinear_API[PyBobLearnLinearWCCNTrainer_Check_NUM] = (void *)&PyBobLearnLinearWCCNTrainer_Check;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyBobLearnLinear_API,
      BOB_EXT_MODULE_PREFIX "." BOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyBobLearnLinear_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(module, "_C_API", c_api_object);

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;
  if (import_bob_learn_activation() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
