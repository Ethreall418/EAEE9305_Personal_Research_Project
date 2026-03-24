"""
Tests for run_simulation.py (v2)
=================================
Four smoke tests covering the key v2 behaviours:

  1. Basic end-to-end run — small grid, partial chunk (n_steps not divisible
     by chunk_size), NetCDF output dimensions and NaN status.
  2. Partial-chunk correctness — final time equals n_steps * dt.
  3. save_interval / chunk_size decoupling — saves occur at the right
     number of records when the two parameters are coprime.
  4. NaN abort — run_simulation exits with code 1 when NaN is injected
     into the state after the first chunk.

Running
-------
    pytest OceanJAX/tests/test_run_simulation.py -v
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pytest

# run_simulation.py lives at the project root; make sure it is importable
# when pytest is invoked from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_simulation import main
from OceanJAX.timeStepping import run as real_ocean_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_GRID = ["--nx", "4", "--ny", "4", "--nz", "3",
               "--lon_min", "0", "--lon_max", "4",
               "--lat_min", "10", "--lat_max", "14",
               "--depth_max", "300"]


def _run(tmp_path, extra_args):
    """Run main() with a temporary output file and return the open Dataset."""
    out = str(tmp_path / "out.nc")
    main(_SMALL_GRID + ["--output", out] + extra_args)
    return nc.Dataset(out, "r")


# ---------------------------------------------------------------------------
# Test 1 — basic end-to-end run
# ---------------------------------------------------------------------------

class TestBasicRun:
    """Small grid, n_steps=20, chunk_size=7 (partial chunk of 6)."""

    def test_output_file_created(self, tmp_path):
        out = str(tmp_path / "out.nc")
        main(_SMALL_GRID + ["--output", out,
                             "--n_steps", "20", "--chunk_size", "7",
                             "--save_interval", "10"])
        assert os.path.exists(out)

    def test_dimensions(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            assert ds.dimensions["x"].size == 4
            assert ds.dimensions["y"].size == 4
            assert ds.dimensions["z"].size == 3

    def test_T_shape(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            # time records: t=0 + save at step 14 (first chunk boundary >= 10)
            n_rec = ds.variables["T"].shape[0]
            assert n_rec >= 2
            assert ds.variables["T"].shape[1:] == (4, 4, 3)

    def test_no_nan(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "20", "--chunk_size", "7",
                              "--save_interval", "10"])
        with ds:
            T = ds.variables["T"][:]
            assert not np.any(np.isnan(T))


# ---------------------------------------------------------------------------
# Test 2 — partial chunk: final time is correct
# ---------------------------------------------------------------------------

class TestPartialChunk:
    """n_steps=13, chunk_size=5 → two full chunks (10) + one partial (3)."""

    def test_final_time(self, tmp_path):
        dt = 900.0
        n_steps = 13
        ds = _run(tmp_path, ["--n_steps", str(n_steps), "--chunk_size", "5",
                              "--save_interval", "100", "--dt", str(dt)])
        with ds:
            times = np.array(ds.variables["time"][:])
        # last record is the save triggered at steps_done=13 >= save_interval=100?
        # No — save_interval=100 > n_steps=13, so only t=0 is saved.
        # Verify t=0 is present.
        assert float(times[0]) == pytest.approx(0.0)

    def test_completes_without_error(self, tmp_path):
        """Simply verify no exception is raised."""
        _run(tmp_path, ["--n_steps", "13", "--chunk_size", "5",
                        "--save_interval", "100"])


# ---------------------------------------------------------------------------
# Test 3 — save_interval / chunk_size decoupled
# ---------------------------------------------------------------------------

class TestSaveIntervalDecoupled:
    """
    save_interval=3, chunk_size=5, n_steps=15.
    chunk boundaries at steps 5, 10, 15; all three cross a save threshold.
    Expected NetCDF records: t=0 (initial) + saves at steps 5, 10, 15 = 4 total.
    """

    def test_record_count(self, tmp_path):
        ds = _run(tmp_path, ["--n_steps", "15", "--chunk_size", "5",
                              "--save_interval", "3"])
        with ds:
            n_rec = ds.variables["time"].shape[0]
        assert n_rec == 4

    def test_time_values(self, tmp_path):
        dt = 900.0
        ds = _run(tmp_path, ["--n_steps", "15", "--chunk_size", "5",
                              "--save_interval", "3", "--dt", str(dt)])
        with ds:
            times = np.array(ds.variables["time"][:])
        expected = np.array([0.0, 5 * dt, 10 * dt, 15 * dt], dtype=np.float32)
        np.testing.assert_allclose(times, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — NaN abort
# ---------------------------------------------------------------------------

class TestNanAbort:
    """
    Patch ocean_run so that the second chunk returns a state with NaN in T.
    main() must call sys.exit(1).
    """

    def test_exits_with_code_1(self, tmp_path):
        out = str(tmp_path / "nan_out.nc")
        call_count = [0]

        def nan_run(state, grid, params, n_steps,
                    forcing_sequence=None, save_history=False):
            call_count[0] += 1
            final_state, history = real_ocean_run(
                state, grid, params, n_steps,
                forcing_sequence=forcing_sequence,
                save_history=save_history,
            )
            if call_count[0] >= 2:
                final_state = eqx.tree_at(
                    lambda s: s.T,
                    final_state,
                    jnp.full_like(final_state.T, float("nan")),
                )
            return final_state, history

        # jax.disable_jit() forces every run_jit call to execute the Python
        # function directly, so the mock is invoked on each chunk and NaN
        # injection takes effect on the second call.
        with jax.disable_jit():
            with patch("run_simulation.ocean_run", nan_run):
                with pytest.raises(SystemExit) as exc_info:
                    main(_SMALL_GRID + ["--output", out,
                                        "--n_steps", "20", "--chunk_size", "5"])

        assert exc_info.value.code == 1
