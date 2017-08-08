from distutils.core import setup


setup(
    name='wfns',
    version='0.0',
    description='Generalized wavefunction package.',
    url='',
    license='MIT',
    author='Ayers Group',
    author_email='',
    package_dir={'wfns': 'wfns'},
    packages=['wfns',
              'wfns.backend',
              'wfns.hamiltonian',
              'wfns.solver',
              'wfns.wavefunction', 'wfns.wavefunction.ci', 'wfns.wavefunction.ci', 'wfns.wavefunction.geminals', 'wfns.wavefunction.nonorth',
              'wfns.wrapper'],
    # test_suite='nose.collector',
    requires=['numpy', 'scipy', 'gmpy2'],
)
