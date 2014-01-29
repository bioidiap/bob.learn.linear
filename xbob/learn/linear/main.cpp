/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 13 Dec 2013 12:35:59 CET
 *
 * @brief Bindings to bob::machine
 */

#define XBOB_LEARN_LINEAR_MODULE
#include <xbob.learn.linear/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <xbob.learn.activation/api.h>
#include <xbob.blitz/cleanup.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::machine's linear machine and trainers");

int PyXbobLearnLinear_APIVersion = XBOB_LEARN_LINEAR_API_VERSION;

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  XBOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods, 
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

  PyBobLearnLinearMachine_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearMachine_Type) < 0) return 0;

  PyBobLearnLinearPCATrainer_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearPCATrainer_Type) < 0) return 0;

  PyBobLearnLinearFisherLDATrainer_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearFisherLDATrainer_Type) < 0) return 0;

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

  /* register some constants */
  if (PyModule_AddIntConstant(m, "__api_version__", XBOB_IO_API_VERSION) < 0) return 0;
  if (PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION) < 0) return 0;

  /* register the types to python */
  Py_INCREF(&PyBobLearnLinearMachine_Type);
  if (PyModule_AddObject(m, "Machine", (PyObject *)&PyBobLearnLinearMachine_Type) < 0) return 0;

  Py_INCREF(&PyBobLearnLinearPCATrainer_Type);
  if (PyModule_AddObject(m, "PCATrainer", (PyObject *)&PyBobLearnLinearPCATrainer_Type) < 0) return 0;

  Py_INCREF(&PyBobLearnLinearFisherLDATrainer_Type);
  if (PyModule_AddObject(m, "FisherLDATrainer", (PyObject *)&PyBobLearnLinearFisherLDATrainer_Type) < 0) return 0;

  static void* PyXbobLearnLinear_API[PyXbobLearnLinear_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyXbobLearnLinear_API[PyXbobLearnLinear_APIVersion_NUM] = (void *)&PyXbobLearnLinear_APIVersion;

  /******************************************
   * Bindings for xbob.learn.linear.Machine *
   ******************************************/

  PyXbobLearnLinear_API[PyBobLearnLinearMachine_Type_NUM] = (void *)&PyBobLearnLinearMachine_Type;

  PyXbobLearnLinear_API[PyBobLearnLinearMachine_Check_NUM] = (void *)&PyBobLearnLinearMachine_Check;

  PyXbobLearnLinear_API[PyBobLearnLinearMachine_NewFromSize_NUM] = (void *)&PyBobLearnLinearMachine_NewFromSize;

  /*********************************************
   * Bindings for xbob.learn.linear.PCATrainer *
   *********************************************/

  PyXbobLearnLinear_API[PyBobLearnLinearPCATrainer_Type_NUM] = (void *)&PyBobLearnLinearPCATrainer_Type;

  PyXbobLearnLinear_API[PyBobLearnLinearPCATrainer_Check_NUM] = (void *)&PyBobLearnLinearPCATrainer_Check;

  /***************************************************
   * Bindings for xbob.learn.linear.FisherLDATrainer *
   ***************************************************/

  PyXbobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Type_NUM] = (void *)&PyBobLearnLinearFisherLDATrainer_Type;

  PyXbobLearnLinear_API[PyBobLearnLinearFisherLDATrainer_Check_NUM] = (void *)&PyBobLearnLinearFisherLDATrainer_Check;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyXbobLearnLinear_API,
      XBOB_EXT_MODULE_PREFIX "." XBOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyXbobLearnLinear_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports xbob.learn.activation C-API + dependencies */
  if (import_xbob_blitz() < 0) return 0;
  if (import_xbob_io() < 0) return 0;
  if (import_xbob_learn_activation() < 0) return 0;

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
