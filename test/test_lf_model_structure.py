"""Tests for _build_model_structure and _adjust_params_for_redshift.

These tests do NOT require XSPEC.  They work by:

* Layer 1 — pure-function component parsing with fake component lists.
* Layer 2 — extracting the ``RedshiftTriggerExtrapolator`` class via
  ``exec`` with mocked ``cosmo``, then testing the real
  ``_adjust_params_for_redshift`` method via ``__new__``.
"""

import re
import dataclasses
import pytest
from typing import List, Dict, Optional, Tuple
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock cosmo (Planck18-like) — only injected into the exec namespace below,
# never into sys.modules, so it cannot pollute other tests.
# ---------------------------------------------------------------------------
def _mock_luminosity_distance(z):
    return type("Q", (), {"value": 1000.0 * (1.0 + z) ** 1.5})()


def _mock_comoving_distance(z):
    return type("Q", (), {"value": 1000.0 * (1.0 + z) ** 0.5})()


# ---------------------------------------------------------------------------
# Helpers: extract pure functions + class from source
# ---------------------------------------------------------------------------
def _load_module_objects():
    """Return (_build_model_structure, _ModelStructure, RedshiftTriggerExtrapolator)."""
    content = open("src/jinwu/lf/redshift.py").read()

    # Pure functions (before class definition)
    match = re.search(
        r"_XSPEC_COMPONENT_ALIASES = .*?(?=\nclass RedshiftTriggerExtrapolator)",
        content,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Cannot extract pure-function layer")
    ns = {
        "dataclasses": dataclasses,
        "List": List,
        "Dict": Dict,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(match.group(0), ns)

    # Class code (everything from 'class RedshiftTriggerExtrapolator')
    class_code = content[content.index("class RedshiftTriggerExtrapolator"):]
    # Build a mock cosmo that matches astropy's Planck18 API but does not
    # require scipy.  Injected *only* into the exec namespace — never into
    # sys.modules.
    _mock_cosmo = MagicMock()
    _mock_cosmo.luminosity_distance = _mock_luminosity_distance
    _mock_cosmo.comoving_distance = _mock_comoving_distance

    exec_ns = {
        "cosmo": _mock_cosmo,
        "XSPEC_COSMO_PLANCK18": "67.66 -0.534016305544544 0.6888463055445441",
        "np": __import__("numpy"),
        "XspecKFactory": MagicMock(),
        "TriggerDecider": MagicMock(),
        "BackgroundSimple": MagicMock(),
        "LightcurveSNREvaluator": MagicMock(),
        "BackgroundPrior": MagicMock(),
        "BackgroundCountsPosterior": MagicMock(),
        "Optional": Optional,
        "Tuple": Tuple,
        "Dict": Dict,
        "List": List,
        "Any": object,
        "Path": type(__import__("pathlib").Path()),
        "os": __import__("os"),
        "dataclasses": dataclasses,
        "TYPE_CHECKING": False,
    }
    exec(class_code, exec_ns)

    return (
        ns["_build_model_structure"],
        ns["_ModelStructure"],
        exec_ns["RedshiftTriggerExtrapolator"],
    )


_build_model_structure, _ModelStructure, RedshiftTriggerExtrapolator = (
    _load_module_objects()
)


# ===================================================================
# Layer 1 — component detection and parameter indexing
# ===================================================================
class TestBuildModelStructure:
    def test_powerlaw_structure(self):
        ms = _build_model_structure(
            "tbabs*ztbabs*powerlaw",
            (0.5, 0.3, 0.01, 1.8, 1e-4),
            ["TBabs", "zTBabs", "powerlaw"],
            {
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        assert ms.has_tbabs and ms.has_ztbabs
        assert ms.has_powerlaw and not ms.has_zpowerlw
        assert not ms.has_cflux and not ms.has_clumin
        assert ms.tbabs_nH_idx == 0
        assert ms.ztbabs_redshift_idx == 2
        assert ms.phoindex_param_idx == 3
        assert ms.norm_param_idx == 4
        assert ms.phoindex_base == 1.8
        assert ms.norm0_base == 1e-4
        assert ms.all_redshift_indices == (2,)
        assert ms.is_powerlaw_model() and not ms.is_zpowerlw_model()

    def test_zpowerlw_structure(self):
        ms = _build_model_structure(
            "tbabs*ztbabs*zpowerlw",
            (0.5, 0.3, 0.01, 1.8, 0.01, 1e-4),
            ["TBabs", "zTBabs", "zpowerlw"],
            {
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "zpowerlw": ["PhoIndex", "Redshift", "norm"],
            },
        )
        assert ms.has_zpowerlw and not ms.has_powerlaw
        assert ms.ztbabs_redshift_idx == 2
        assert ms.zpowerlw_redshift_idx == 4
        assert ms.phoindex_param_idx == 3
        assert ms.norm_param_idx == 5
        assert ms.norm0_base == 1e-4
        assert ms.phoindex_base == 1.8
        assert ms.all_redshift_indices == (2, 4)
        assert not ms.is_powerlaw_model() and ms.is_zpowerlw_model()

    def test_cflux_structure(self):
        ms = _build_model_structure(
            "tbabs*ztbabs*cflux*powerlaw",
            (0.5, 0.3, 0.01, 0.5, 4.0, -11.0, 1.8, 1.0),
            ["TBabs", "zTBabs", "cflux", "powerlaw"],
            {
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "cflux": ["Emin", "Emax", "lg10Flux"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        assert ms.has_cflux
        assert ms.cflux_lg10flux_idx == 5
        assert ms.cflux_emin_idx == 3
        assert ms.cflux_emax_idx == 4
        assert ms.lg10flux_base == -11.0
        assert ms.norm_param_idx == 7
        assert ms.phoindex_param_idx == 6
        assert ms.all_redshift_indices == (2,)

    def test_clumin_structure(self):
        ms = _build_model_structure(
            "tbabs*ztbabs*clumin*powerlaw",
            (0.5, 0.3, 0.01, 0.5, 4.0, 42.0, 0.01, 1.8, 1.0),
            ["TBabs", "zTBabs", "clumin", "powerlaw"],
            {
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "clumin": ["Emin", "Emax", "lg10Lum", "Redshift"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        assert ms.has_clumin
        assert ms.clumin_lg10lumin_idx == 5
        assert ms.clumin_redshift_idx == 6
        assert ms.lg10lumin_base == 42.0
        assert ms.all_redshift_indices == (2, 6)

    def test_clumin_lg10Lumin_variant(self):
        ms = _build_model_structure(
            "tbabs*ztbabs*clumin*powerlaw",
            (0.5, 0.3, 0.01, 0.5, 4.0, 42.0, 0.01, 1.8, 1.0),
            ["TBabs", "zTBabs", "clumin", "powerlaw"],
            {
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "clumin": ["Emin", "Emax", "lg10Lumin", "Redshift"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        assert ms.clumin_lg10lumin_idx == 5
        assert ms.lg10lumin_base == 42.0

    def test_case_insensitive_components(self):
        ms = _build_model_structure(
            "TBabs*ZTBabs*Powerlaw",
            (0.5, 0.3, 0.01, 1.8, 1e-4),
            ["TBabs", "ZTBabs", "Powerlaw"],
            {
                "TBabs": ["nH"],
                "ZTBabs": ["nH", "Redshift"],
                "Powerlaw": ["PhoIndex", "norm"],
            },
        )
        assert ms.has_tbabs and ms.has_ztbabs and ms.has_powerlaw

    def test_unrecognised_model_still_finds_norm(self):
        """Unrecognised components should still record the last norm."""
        ms = _build_model_structure(
            "bbodyrad",
            (1.0, 1e-3),
            ["bbodyrad"],
            {"bbodyrad": ["kT", "norm"]},
        )
        assert ms.has_powerlaw is False
        assert ms.has_zpowerlw is False
        assert ms.has_cflux is False
        assert ms.has_clumin is False
        assert ms.all_redshift_indices == ()
        assert ms.norm_param_idx == 1
        assert ms.norm0_base == 1e-3


# ===================================================================
# Layer 2 — test the REAL _adjust_params_for_redshift
# ===================================================================
class TestAdjustParamsForRedshift:
    """Lightweight instances via ``__new__`` test the real method."""

    @staticmethod
    def _make_extrapolator(model_expr, z0, params, comp_names, comp_params):
        ms = _build_model_structure(model_expr, params, comp_names, comp_params)
        obj = RedshiftTriggerExtrapolator.__new__(RedshiftTriggerExtrapolator)
        obj._model_structure = ms
        obj.z0 = float(z0)
        obj.params = tuple(params)
        return obj

    def test_identity_at_same_z(self):
        obj = self._make_extrapolator(
            "tbabs*ztbabs*powerlaw",
            z0=0.01,
            params=(0.5, 0.3, 0.01, 1.8, 1e-4),
            comp_names=["TBabs", "zTBabs", "powerlaw"],
            comp_params={
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        result = obj._adjust_params_for_redshift(0.01)
        assert result[2] == 0.01
        assert result[4] == pytest.approx(1e-4)

    def test_powerlaw_adjustment(self):
        obj = self._make_extrapolator(
            "tbabs*ztbabs*powerlaw",
            z0=0.01,
            params=(0.5, 0.3, 0.01, 1.8, 1e-4),
            comp_names=["TBabs", "zTBabs", "powerlaw"],
            comp_params={
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "powerlaw": ["PhoIndex", "norm"],
            },
        )
        result = obj._adjust_params_for_redshift(1.0)
        assert result[2] == 1.0
        # Mock cosmo: dL = 1000*(1+z)^1.5
        dL0 = 1000 * 1.01**1.5
        dL1 = 1000 * 2.0**1.5
        expected = 1e-4 * (dL0 / dL1) ** 2 * (2.0 / 1.01) ** (2 - 1.8)
        assert result[4] == pytest.approx(expected)

    def test_zpowerlw_raises(self):
        obj = self._make_extrapolator(
            "tbabs*ztbabs*zpowerlw",
            z0=0.01,
            params=(0.5, 0.3, 0.01, 1.8, 0.01, 1e-4),
            comp_names=["TBabs", "zTBabs", "zpowerlw"],
            comp_params={
                "TBabs": ["nH"],
                "zTBabs": ["nH", "Redshift"],
                "zpowerlw": ["PhoIndex", "Redshift", "norm"],
            },
        )
        with pytest.raises(NotImplementedError, match="zpowerlw"):
            obj._adjust_params_for_redshift(1.0)

    def test_unrecognised_raises(self):
        ms = _ModelStructure(model_expr="unknown")
        obj = RedshiftTriggerExtrapolator.__new__(RedshiftTriggerExtrapolator)
        obj._model_structure = ms
        obj.z0 = 0.01
        obj.params = (1.0,)
        with pytest.raises(ValueError, match="Cannot adjust"):
            obj._adjust_params_for_redshift(1.0)



# ===================================================================
# Layer 3 — formula arithmetic (documentation / sanity)
# ===================================================================
class TestFormulaArithmetic:
    def test_powerlaw_identity(self):
        n0, g = 1e-4, 2.0
        norm = n0 * (1000 / 1000) ** 2 * (1.01 / 1.01) ** (2 - g)
        assert norm == pytest.approx(n0)

    def test_powerlaw_dimmed(self):
        n0, g = 1e-4, 2.0
        norm = n0 * 0.25 * ((2.0 / 1.01) ** 0)
        assert norm == pytest.approx(2.5e-5)

    def test_spectral_term(self):
        n0, g = 1e-4, 3.0
        norm = n0 * 0.25 * (1.01 / 2.0)
        assert norm == pytest.approx(1.2625e-5)
