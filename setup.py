"""
Distutils' setup file.

"""

from distutils.core import setup

setup(
    name="Geminals",
    version="0.0.1",
    url="https://github.com/quantumelephant/olsens",
    packages=["geminals", "geminals.test"],
    license="LICENSE",
    description="Geminal and geminal-like wavefunctions HORTON module.",
    install_requires=[
        "numpy",
        "scipy",
        "horton"
        "romin",
    ],
)

# vim: set textwidth=90 :
