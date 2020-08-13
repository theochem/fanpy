"""Test fanpy.script.utils."""
import pytest
from fanpy.scripts.utils import check_inputs, parser


def test_check_inputs():
    "Test fanpy.scripts.utils.check_inputs."
    with pytest.raises(TypeError):
        check_inputs(
            2.0, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0
        )
    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            ham_noise='0'
        )
    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            wfn_noise='0'
        )
    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            optimize_orbs=None
        )
    with pytest.raises(TypeError):
        check_inputs(
            2, 1, "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy\n", "ap1rog", [1, 2], "projected", "least_squares", 0.0
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "test", [1, 2], "projected", "least_squares", 0.0
        )
    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2.0], "projected", "least_squares", 0.0
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [0, 2], "projected", "least_squares", 0.0
        )

    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "test", "least_squares", 0.0
        )

    with pytest.raises(ValueError):
        check_inputs(2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "test", 0.0)

    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "one_energy", "least_square", 0.0
        )
    with pytest.raises(ValueError):
        check_inputs(2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "cma", 0.0)
    with pytest.raises(ValueError):
        check_inputs(2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "diag", 0.0)

    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "fci", [1, 2], "projected", "diag", 0.0,
            optimize_orbs=True
        )

    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            load_orbs=1
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            filename="sdf;"
        )

    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            memory=10
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            memory="10bytes"
        )

    with pytest.raises(TypeError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            solver_kwargs=1
        )
    with pytest.raises(ValueError):
        check_inputs(
            2, "oneint.npy", "twoint.npy", "ap1rog", [1, 2], "projected", "least_squares", 0.0,
            wfn_kwargs="sdfdsf;"
        )


def test_parser():
    "Test fanpy.scripts.utils.parser."
    args = parser.parse_args([
        "--nelec", "1", "--one_int_file", "oneint.npy", "--two_int_file", "twoint.npy",
        "--wfn_type", "ap1rog"
    ])
    assert args.nelec == 1
    assert args.one_int_file == "oneint.npy"
    assert args.two_int_file == "twoint.npy"
    assert args.wfn_type == "ap1rog"
    assert args.nuc_nuc == 0
    assert isinstance(args.optimize_orbs, bool) and not args.optimize_orbs
    assert args.pspace_exc == [1, 2]
    assert args.objective == "projected"
    assert args.solver == "least_squares"
    assert args.ham_noise == 0
    assert args.wfn_noise == 0
    assert args.solver_kwargs is None
    assert args.load_orbs is None
    assert args.load_ham is None
    assert args.load_ham_um is None
    assert args.load_wfn is None
    assert args.save_chk == ""
    assert args.memory is None
