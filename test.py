import numpy as np

from prowl import horton as ht, permanent

from apig import Apig
from ap1rog import Ap1rog

n = 6
h = ht.ap1rog(n=n, fn="test/li2.xyz", x="apig", basis="sto-3g")
p = n // 2
H = h['H']
G = h['G']
k = H.shape[0]
