#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core', 'bob.math', 'bob.io.base', 'bob.learn.activation']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['boost']
boost_modules = ['system']

setup(

    name='bob.learn.linear',
    version=version,
    description='Linear Machine and Trainers for Bob',
    url='http://gitlab.idiap.ch/bob/bob.learn.linear',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,



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
          "bob/learn/linear/cpp/bic.cpp",
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
          "bob/learn/linear/bic.cpp",
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
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

)
