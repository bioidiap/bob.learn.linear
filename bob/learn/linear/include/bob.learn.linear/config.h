/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 13 Dec 2013 11:50:29 CET
 *
 * @brief General directives for all modules in bob.learn.linear
 */

#ifndef BOB_LEARN_LINEAR_CONFIG_H
#define BOB_LEARN_LINEAR_CONFIG_H

/* Macros that define versions and important names */
#define BOB_LEARN_LINEAR_API_VERSION 0x0201

#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>

  /**
   * bob.learn.linear c/c++ api version
   */
  static PyObject* bob_learn_linear_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_LEARN_LINEAR_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION

#endif /* BOB_LEARN_LINEAR_CONFIG_H */
