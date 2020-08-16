from fanpy.wfn.geminal.apg4 import APG4


class APG7(APG4):
    def __init__(
        self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None, tol=1e-4
    ):
        super().__init__(
            nelec,
            nspin,
            dtype=dtype,
            memory=memory,
            ngem=ngem,
            orbpairs=orbpairs,
            params=params,
            tol=tol,
            num_matchings=1,
        )
