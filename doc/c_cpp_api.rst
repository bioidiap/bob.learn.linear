.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

The C++ API of ``bob.learn.linear`` allows users to leverage from automatic
converters for classes in :py:mod:`bob.learn.linear`.  To use the C API,
clients should first, include the header file ``<bob.learn.linear/api.h>`` on
their compilation units and then, make sure to call once
``import_bob_learn_linear()`` at their module instantiation, as explained at
the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c++

   #include <bob.learn.linear/api.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject* m Py_InitModule("client", ClientMethods);

     if (!m) return 0;

     // imports dependencies
     if (import_bob_blitz() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     if (import_bob_io_base() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     if (import_bob_learn_activation() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     if (import_bob_learn_linear() < 0) {
       PyErr_Print();
       PyErr_SetString(PyExc_ImportError, "cannot import module");
       return 0;
     }

     return m;

   }


Linear Machines
---------------

.. todo:: Write this section

