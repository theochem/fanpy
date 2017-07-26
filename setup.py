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
    packages=['wfns'],
    # test_suite='nose.collector',
    requires=['numpy', 'scipy', 'gmpy2'],
)
