#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core', 'bob.math', 'bob.io.base', 'bob.learn.activation']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

packages = ['boost']
boost_modules = ['system']

version = '2.0.0a0'

setup(

    name='bob.learn.linear',
    version=version,
    description='Bindings for bob.machine\'s LinearMachine and Trainers',
    url='http://github.com/bioidiap/bob.learn.linear',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.core',
      'bob.io.base',
      'bob.math',
      'bob.learn.activation',
    ],

    namespace_packages=[
      "bob",
      "bob.learn",
    ],

    ext_modules = [
      Extension("bob.learn.linear.version",
        [
          "bob/learn/linear/version.cpp",
        ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Library("bob.learn.linear.bob_learn_linear",
        [
          "bob/learn/linear/cpp/machine.cpp",
          "bob/learn/linear/cpp/pca.cpp",
          "bob/learn/linear/cpp/lda.cpp",
          "bob/learn/linear/cpp/logreg.cpp",
          "bob/learn/linear/cpp/whitening.cpp",
          "bob/learn/linear/cpp/wccn.cpp",
        ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Extension("bob.learn.linear._library",
        [
          "bob/learn/linear/machine.cpp",
          "bob/learn/linear/pca.cpp",
          "bob/learn/linear/lda.cpp",
          "bob/learn/linear/logreg.cpp",
          "bob/learn/linear/whitening.cpp",
          "bob/learn/linear/wccn.cpp",
          "bob/learn/linear/main.cpp",
          ],
        bob_packages = bob_packages,
        version = version,
        packages = packages,
        boost_modules = boost_modules,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    entry_points={
      'console_scripts': [
      ],
    },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
