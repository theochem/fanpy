"""Setup script for installing the package."""
import codecs
from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='wfns',
      version='0.0.0',
      description='Package for solving the Schrodinger equation.',
      long_description=long_description,
      url='https://github.com/quantumelephant/fanpy',
      license='GNU Version 3',
      author='Taewon D. Kim',
      author_email='kimt33@mcmaster.ca',
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers of new methods for solving the Schrodinger '
                   'equation',
                   'Topic :: Method Development :: Schrodinger equation',
                   'License :: OSI Approved :: GNU Version 3',
                   'Programming Language :: Python :: 3.6'],
      keywords='wavefunction hamiltonian optimization',
      package_dir={'wfns': 'wfns'},
      packages=['wfns', 'wfns.backend', 'wfns.ham',
                'wfns.objective', 'wfns.objective.schrodinger', 'wfns.objective.constraints',
                'wfns.solver',
                'wfns.wfn', 'wfns.wfn.ci', 'wfns.wfn.composite', 'wfns.wfn.geminal',
                'wfns.wrapper'],
      # test_suite='nose.collector',
      install_requires=['numpy', 'scipy', 'gmpy2'],
      extras_require={'horton': [], 'pyscf': []},
      tests_requires=['nose', 'cardboardlint'],
      package_data={},
      data_files=[],
      )
