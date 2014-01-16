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
#include <xbob.blitz/capi.h>
#include <xbob.io/api.h>
#include <xbob.learn.activation/api.h>

static PyMethodDef library_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(library_docstr, "bob::machine's linear machine and trainers");

int PyXbobLearnLinear_APIVersion = XBOB_LEARN_LINEAR_API_VERSION;

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {

  PyBobLearnLinearMachine_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearMachine_Type) < 0) return;

  PyBobLearnLinearPCATrainer_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearPCATrainer_Type) < 0) return;

  PyBobLearnLinearFisherLDATrainer_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobLearnLinearFisherLDATrainer_Type) < 0) return;

  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, library_methods, library_docstr);

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_LEARN_LINEAR_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION);

  /* register the types to python */
  Py_INCREF(&PyBobLearnLinearMachine_Type);
  PyModule_AddObject(m, "Machine", (PyObject *)&PyBobLearnLinearMachine_Type);

  Py_INCREF(&PyBobLearnLinearPCATrainer_Type);
  PyModule_AddObject(m, "PCATrainer", (PyObject *)&PyBobLearnLinearPCATrainer_Type);

  Py_INCREF(&PyBobLearnLinearFisherLDATrainer_Type);
  PyModule_AddObject(m, "FisherLDATrainer", (PyObject *)&PyBobLearnLinearFisherLDATrainer_Type);

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

  /* imports the NumPy C-API */
  import_array();

  /* imports xbob.blitz C-API */
  import_xbob_blitz();

  /* imports xbob.io C-API */
  import_xbob_io();

  /* imports xbob.learn.activation C-API */
  import_xbob_learn_activation();

}
