#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.io.base', 'bob.learn.activation']))
from bob.blitz.extension import Extension
import bob.io.base
import bob.learn.activation

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'bob', 'learn', 'linear', 'include')
include_dirs = [
    package_dir,
    bob.blitz.get_include(),
    bob.io.base.get_include(),
    bob.learn.activation.get_include()
    ]

packages = ['bob-machine >= 2.0.0a2', 'bob-io >= 2.0.0a2']
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

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.io.base',
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
        packages = packages,
        include_dirs = include_dirs,
        version = version,
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
          "bob/learn/linear/cpp/machine.cpp",
          "bob/learn/linear/cpp/pca.cpp",
          "bob/learn/linear/cpp/lda.cpp",
          "bob/learn/linear/cpp/logreg.cpp",
          "bob/learn/linear/cpp/whitening.cpp",
          "bob/learn/linear/cpp/wccn.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      ],

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
