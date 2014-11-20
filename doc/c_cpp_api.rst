.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

.. todo:: Correct the C++ API (it seems to be a copy of bob.learn.activation)

The C++ API of ``bob.learn.linear`` allows users to leverage from automatic
converters for classes in :py:class:`bob.learn.linear`.  To use the C API,
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

     if (import_bob_blitz() < 0) return 0;
     if (import_bob_io() < 0) return 0;
     if (import_bob_learn_activation() < 0) return 0;

     return m;

   }


Activation Functors
-------------------

.. cpp:type:: PyBobMachineActivationObject

   The pythonic object representation for a ``bob::machine::Activation``
   object. It is the base class of all activation functors available in
   |project|. In C/C++ code, we recommend you only manipulate objects like this
   to keep your code agnostic to the activation type being used.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        bob::machine::Activation* base;
      } PyBobMachineActivationObject;

   .. cpp:member:: bob::machine::Activation* base

      A pointer to the activation functor virtual implementation.


.. cpp:function:: int PyBobMachineActivation_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBobMachineActivationObject``.
   Returns ``1`` if it is, and ``0`` otherwise.


.. note::

   Other object definitions exist for each of the specializations for
   activation functors found in |project|. They are exported through the module
   C-API, but we don't recommend using them since you'd loose generality. In
   case you do absolutely need to use any of these derivations, they have all
   the same object configuration:

   .. code-block:: c++

      typedef struct {
        PyBobMachineActivationObject parent;
        bob::machine::<Subtype>Activation* base;
      } PyBobMachine<Subtype>ActivationObject;

   Presently, ``<Subtype>`` can be one of:

     * Identity
     * Linear
     * Logistic
     * HyperbolicTangent
     * MultipliedHyperbolicTangent

   Type objects are also named consistently like
   ``PyBobMachine<Subtype>_Type``.

.. include:: links.rst
