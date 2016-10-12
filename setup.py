from __future__ import absolute_import, division, print_function

from distutils.core import setup


if __name__ == "__main__":

    setup(
        name="geminals",
        packages=[
            "geminals",
            "geminals.proj",
            "geminals.ci",
            ],
        url="https://github.com/QuantumElephant/olsens",
        )
