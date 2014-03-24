.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 13 Dec 2013 12:35:22 CET

.. image:: https://travis-ci.org/bioidiap/xbob.learn.linear.svg?branch=master
   :target: https://travis-ci.org/bioidiap/xbob.learn.linear
.. image:: https://coveralls.io/repos/bioidiap/xbob.learn.linear/badge.png
   :target: https://coveralls.io/r/bioidiap/xbob.learn.linear
.. image:: http://img.shields.io/github/tag/bioidiap/xbob.learn.linear.png
   :target: https://github.com/bioidiap/xbob.learn.linear
.. image:: http://img.shields.io/pypi/v/xbob.learn.linear.png
   :target: https://pypi.python.org/pypi/xbob.learn.linear
.. image:: http://img.shields.io/pypi/dm/xbob.learn.linear.png
   :target: https://pypi.python.org/pypi/xbob.learn.linear

==============================================================
 Python bindings for bob.machine's LinearMachine and Trainers
==============================================================

This package contains a set of Pythonic bindings for Bob's LinearMachine and
Trainers.

Installation
------------

Install it through normal means, via PyPI or use ``zc.buildout`` to bootstrap
the package and run test units.

Documentation
-------------

You can generate the documentation for this package, after installation, using
Sphinx::

  $ sphinx-build -b html doc sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.

Testing
-------

You can run a set of tests using the nose test runner::

  $ nosetests -sv xbob.learn.linear

.. warning::

   If Bob <= 1.2.1 is installed on your python path, nose will automatically
   load the old version of the insulate plugin available in Bob, which will
   trigger the loading of incompatible shared libraries (from Bob itself), in
   to your working binary. This will cause a stack corruption. Either remove
   the centrally installed version of Bob, or build your own version of Python
   in which Bob <= 1.2.1 is not installed.

You can run our documentation tests using sphinx itself::

  $ sphinx-build -b doctest doc sphinx

You can test overall test coverage with::

  $ nosetests --with-coverage --cover-package=xbob.learn.linear

The ``coverage`` egg must be installed for this to work properly.

Development
-----------

To develop this package, install using ``zc.buildout``, using the buildout
configuration found on the root of the package::

  $ python bootstrap.py
  ...
  $ ./bin/buildout

Tweak the options in ``buildout.cfg`` to disable/enable verbosity and debug
builds.
