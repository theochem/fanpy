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
    packages=['wfns', 'wfns.backend', 'wfns.ham',
              'wfns.objective', 'wfns.objective.schrodinger', 'wfns.objective.constraints',
              'wfns.solver',
              'wfns.wfn', 'wfns.wfn.ci', 'wfns.wfn.composite', 'wfns.wfn.geminal',
              'wfns.wrapper'],
    # test_suite='nose.collector',
    requires=['numpy', 'scipy', 'gmpy2'],
)
