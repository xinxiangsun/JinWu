"""Bayesian X-ray spectral fitting via BXA (Bayesian X-ray Analysis).

This module wraps BXA / UltraNest nested sampling on top of the *existing*
XSPEC session-building helpers in :mod:`jinwu.core.fit`.  It deliberately does
NOT re-implement spectrum loading, model configuration, group linking, or
result generation; those are reused verbatim so the maximum-likelihood (MLE)
and Bayesian (BXA) paths stay in lockstep.

Reused private helpers from ``jinwu.core.fit`` (keep this list in sync if
``fit.py`` changes):

    _require_xspec, _prepared_energy_ranges, _prepared_spectrum_key,
    _prepared_data_groups, _load_prepared_xspec_spectrum,
    _set_prepared_xspec_links, _xspec_model_for_group,
    _configure_prepared_model, _freeze_prepared_parameters,
    _link_default_prepared_model_groups, _count_free_xspec_parameters,
    _generate_xspec_result, _prepared_input_dict, _capture_xspec_log,
    _xspec_show_text, _write_prepared_report, _xspec_spectrum_counts

Key technical constraints honoured here (see the approved plan):

* ``BXASolver.vectorized`` is hard-coded ``False`` and PyXspec is a
  non-reentrant global singleton -> no thread pools / joblib threads / shared
  multiprocessing sessions.  Only process-level (MPI) parallelism is safe.
* ``Fit.statMethod`` must be one of ``cstat``/``cash``/``pstat`` (BXA's
  ``allowed_stats``); anything else raises with a clear message.
* BXA v5 priors use *soft* bounds: ``create_*_prior_for`` require
  ``pmin == pbottom`` and ``ptop == pmax``.  :func:`resolve_priors` unifies
  them before building transformations.
* ``outputfiles_basename`` must be an existing directory; it is created with
  ``mkdir(parents=True, exist_ok=True)`` before the solver is constructed.

All ``bxa`` / ``xspec`` imports are lazy, so importing this module (and
therefore ``jinwu.core``) never requires BXA or HEASoft to be installed.
"""

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from jinwu.core.config import get_fit_settings
from jinwu.core.fit import (
    ModelFitMetrics,
    XRayModelComparisonResult,
    _capture_xspec_log,
    _configure_prepared_model,
    _count_free_xspec_parameters,
    _freeze_prepared_parameters,
    _generate_xspec_result,
    _link_default_prepared_model_groups,
    _load_prepared_xspec_spectrum,
    _prepared_data_groups,
    _prepared_energy_ranges,
    _prepared_input_dict,
    _prepared_spectrum_key,
    _require_xspec,
    _set_prepared_xspec_links,
    _write_prepared_report,
    _xspec_model_for_group,
    _xspec_show_text,
    _xspec_spectrum_counts,
    calculate_bayesian_model_metrics,
    resolve_xray_model_specs,
)

__all__ = [
    "BXAPriorSpec",
    "BXAFitResult",
    "resolve_priors",
    "fit_prepared_bxa",
    "fit_xray_models_bxa",
    "run_bxa_pipeline",
]

#: BXA's ``BXASolver.allowed_stats``.  Poisson likelihoods only.
_ALLOWED_STAT_METHODS: frozenset[str] = frozenset({"cstat", "cash", "pstat"})

_BXA_INSTALL_HINT = (
    "BXA (Bayesian X-ray Analysis) is required for method='bxa'. Install it "
    "with `pip install jinwu[bxa]` (which pulls `bxa` and `ultranest`) inside "
    "an environment that also provides HEASoft/PyXspec, then re-run."
)


def _require_bxa():
    """Return the ``bxa.xspec`` module, raising install guidance if absent."""
    try:
        import bxa.xspec as bxa_xspec
        import bxa.xspec.priors  # noqa: F401  (guarantees priors importable)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_BXA_INSTALL_HINT) from exc
    return bxa_xspec


@dataclass(frozen=True, slots=True)
class BXAPriorSpec:
    """Declarative prior for one thawed XSPEC parameter.

    ``kind`` selects the BXA transformation:

    - ``"uniform"``    -> ``create_uniform_prior_for`` over ``[low, high]``.
    - ``"loguniform"`` -> ``create_loguniform_prior_for`` over ``[low, high]``
      (requires ``low > 0``).
    - ``"gaussian"``   -> ``create_gaussian_prior_for`` with ``mean`` / ``std``.
    - ``"custom"``     -> ``create_custom_prior_for`` with ``transform`` /
      ``aftertransform`` callables.

    When ``low`` / ``high`` are provided they replace the parameter's soft
    bounds; per BXA v5 the hard bounds are unified onto them so that
    ``pmin == pbottom`` and ``ptop == pmax``.  When omitted, the parameter's
    existing soft bounds are used (still unified).
    """

    kind: Literal["uniform", "loguniform", "gaussian", "custom"] = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    transform: Any = None
    aftertransform: Any = None


def _parameter_values_list(parameter) -> list[float]:
    """Return the six XSPEC parameter values as floats.

    ``par.values`` unpacks as ``pval, pdelta, pmin, pbottom, ptop, pmax``.
    Test doubles may expose fewer entries; missing tail values default to 0.
    """
    raw = parameter.values
    values = [float(item) for item in raw]
    if len(values) < 6:
        values.extend([0.0] * (6 - len(values)))
    return values[:6]


def _unify_soft_bounds(parameter, low=None, high=None) -> tuple[float, float]:
    """Force ``pmin == pbottom`` and ``ptop == pmax`` (BXA v5 requirement).

    Optionally override the bottom / top soft limits with ``low`` / ``high``.
    Returns the effective ``(bottom, top)`` used for the prior.
    """
    pval, _pdelta, _pmin, pbottom, ptop, _pmax = _parameter_values_list(parameter)
    bottom = float(low) if low is not None else float(pbottom)
    top = float(high) if high is not None else float(ptop)
    if not (math.isfinite(bottom) and math.isfinite(top)) or top <= bottom:
        raise ValueError(
            f"prior bounds for {parameter.name!r} must satisfy high > low; "
            f"got low={bottom!r}, high={top!r}"
        )
    # value, delta, min=bottom, bottom, top, max=top  (matches fit.py's
    # _set_prepared_parameter_bounds comma-string form).
    parameter.values = f"{pval},,{bottom},{bottom},{top},{top}"
    return bottom, top


def _build_transformation(priors, model, parameter, spec, bottom, top):
    """Dispatch to the appropriate ``bxa.xspec.priors.create_*_prior_for``."""
    kind = spec.kind
    if kind == "uniform":
        return priors.create_uniform_prior_for(model, parameter)
    if kind == "loguniform":
        if bottom <= 0:
            raise ValueError(
                f"loguniform prior for {parameter.name!r} requires low > 0; "
                f"got low={bottom!r}"
            )
        return priors.create_loguniform_prior_for(model, parameter)
    if kind == "gaussian":
        if spec.mean is None or spec.std is None:
            raise ValueError(
                f"gaussian prior for {parameter.name!r} requires mean and std"
            )
        if not math.isfinite(float(spec.std)) or float(spec.std) <= 0:
            raise ValueError(
                f"gaussian prior std for {parameter.name!r} must be positive"
            )
        return priors.create_gaussian_prior_for(
            model, parameter, float(spec.mean), float(spec.std)
        )
    if kind == "custom":
        if spec.transform is None:
            raise ValueError(
                f"custom prior for {parameter.name!r} requires a transform callable"
            )
        aftertransform = spec.aftertransform if spec.aftertransform else (lambda x: x)
        return priors.create_custom_prior_for(
            model, parameter, spec.transform, aftertransform
        )
    raise ValueError(f"Unknown prior kind: {kind!r}")


def resolve_priors(
    xspec,
    model,
    group_models: Sequence[Any],
    prior_specs: Mapping[str, BXAPriorSpec] | None = None,
) -> list[dict[str, Any]]:
    """Build BXA prior transformations for every thawed, unlinked parameter.

    Parameters are discovered exactly as :func:`jinwu.core.fit` counts them
    (``componentNames`` -> ``parameterNames``, skipping frozen or linked
    parameters).  ``prior_specs`` is keyed by ``"component.parameter"``; a
    missing key defaults to a uniform prior over the parameter's existing soft
    bounds.  Each parameter's soft/hard bounds are unified before the
    transformation is created (BXA v5 requirement).
    """
    import bxa.xspec.priors as priors

    specs = dict(prior_specs or {})
    transformations: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for group_index, group_model in enumerate(group_models):
        for component_name in getattr(group_model, "componentNames", ()):
            component = getattr(group_model, component_name, None)
            if component is None:
                continue
            for parameter_name in getattr(component, "parameterNames", ()):
                parameter = getattr(component, parameter_name, None)
                if parameter is None or bool(getattr(parameter, "frozen", False)):
                    continue
                if str(getattr(parameter, "link", "") or "").strip():
                    # Slave parameters tied to a master through ``.link`` are
                    # sampled via the master only -- exactly as
                    # ``_count_free_xspec_parameters`` skips them, so the two
                    # counts stay in lockstep.
                    continue
                # ``prior_specs`` is keyed by the shared ``component.parameter``
                # name so one spec applies to every group.  The *sampling
                # identity* is group-qualified, so deliberately unlinked
                # same-named parameters across groups are each sampled instead
                # of silently de-duplicated (which would leave
                # ``len(paramnames) != free_parameter_count`` and freeze N-1
                # groups at their best-fit value, biasing the posterior/evidence).
                spec_key = f"{component_name}.{parameter_name}"
                identity = (group_index, component_name, parameter_name)
                if identity in seen:
                    continue
                seen.add(identity)
                spec = specs.get(spec_key, BXAPriorSpec())
                bottom, top = _unify_soft_bounds(parameter, spec.low, spec.high)
                transformations.append(
                    _build_transformation(
                        priors, group_model, parameter, spec, bottom, top
                    )
                )
    if not transformations:
        raise RuntimeError(
            "BXA requires at least one thawed, unlinked parameter to sample; "
            "the configured model has none."
        )
    return transformations


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy arrays/scalars into JSON-native containers."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


@dataclass(slots=True)
class BXAFitResult:
    """Result of one BXA nested-sampling fit of a prepared spectrum.

    ``to_dict`` is guaranteed JSON safe: every ``numpy`` array is lowered to a
    nested ``list`` and exotic scalars to Python primitives.
    """

    model_name: str
    paramnames: list[str] = field(default_factory=list)
    logz: float | None = None
    logzerr: float | None = None
    posterior: Any = None
    marginals: Any = None
    best_fit: dict[str, Any] = field(default_factory=dict)
    fit_statistic: float | None = None
    fit_dof: int | None = None
    free_parameter_count: int = 0
    source_counts: float | None = None
    background_counts: float | None = None
    energy_range: dict[str, float] | None = None
    energy_ranges: dict[str, dict[str, float]] = field(default_factory=dict)
    chain_path: str | None = None
    flux_chain: Any = None
    n_live_points: int | None = None
    sampler: str = "ultranest"
    backend: str = "bxa"
    stat_method: str | None = None
    run_settings: dict[str, Any] = field(default_factory=dict)
    parallel_settings: dict[str, Any] = field(default_factory=dict)
    cache_status: str = "overwrite"
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"
    fit_products: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "paramnames": list(self.paramnames),
            "logz": self.logz,
            "logzerr": self.logzerr,
            "posterior": _json_safe(self.posterior),
            "marginals": _json_safe(self.marginals),
            "best_fit": _json_safe(self.best_fit),
            "fit_statistic": self.fit_statistic,
            "fit_dof": self.fit_dof,
            "free_parameter_count": self.free_parameter_count,
            "source_counts": self.source_counts,
            "background_counts": self.background_counts,
            "energy_range": _json_safe(self.energy_range),
            "energy_ranges": _json_safe(self.energy_ranges),
            "chain_path": self.chain_path,
            "flux_chain": _json_safe(self.flux_chain),
            "n_live_points": self.n_live_points,
            "sampler": self.sampler,
            "backend": self.backend,
            "stat_method": self.stat_method,
            "run_settings": _json_safe(self.run_settings),
            "parallel_settings": _json_safe(self.parallel_settings),
            "cache_status": self.cache_status,
            "warnings": list(self.warnings),
            "status": self.status,
            "fit_products": _json_safe(self.fit_products),
        }
        return payload

    def summary(self) -> dict[str, Any]:
        """Compact, human-facing surface (no posterior / chain arrays)."""
        return {
            "model_name": self.model_name,
            "logz": self.logz,
            "logzerr": self.logzerr,
            "fit_statistic": self.fit_statistic,
            "fit_dof": self.fit_dof,
            "free_parameter_count": self.free_parameter_count,
            "paramnames": list(self.paramnames),
            "chain_path": self.chain_path,
            "status": self.status,
        }


def _resolve_prepared_spectra(prepared) -> list[Any]:
    """Normalize ``prepared`` into a list of ready ``PreparedSpectrum``."""
    from jinwu.core.spectrum_prep import PreparedJointSpectrum, PreparedSpectrum

    if isinstance(prepared, PreparedSpectrum):
        prepared_spectra = [prepared]
    elif isinstance(prepared, PreparedJointSpectrum):
        prepared_spectra = list(prepared.spectra)
    elif isinstance(prepared, Sequence) and not isinstance(prepared, (str, bytes)):
        prepared_spectra = list(prepared)
        if not all(
            isinstance(spectrum, PreparedSpectrum) for spectrum in prepared_spectra
        ):
            raise TypeError("prepared sequences must contain PreparedSpectrum items")
    else:
        raise TypeError(
            "prepared must be PreparedSpectrum, PreparedJointSpectrum, "
            "or a sequence of PreparedSpectrum items"
        )
    if not prepared_spectra or any(
        not spectrum.ready for spectrum in prepared_spectra
    ):
        raise RuntimeError("fit_prepared_bxa requires ready prepared spectra")
    if any(spectrum.grouped_pha is None for spectrum in prepared_spectra):
        raise RuntimeError("fit_prepared_bxa requires grouped PHA paths")
    return prepared_spectra


def _resolve_stat_method(stat_method: str | None) -> str:
    """Resolve the None sentinel and enforce BXA's allowed statistics."""
    resolved = (
        stat_method if stat_method is not None else get_fit_settings().statistic
    )
    resolved = str(resolved).lower()
    if resolved not in _ALLOWED_STAT_METHODS:
        raise ValueError(
            f"BXA requires Fit.statMethod in {sorted(_ALLOWED_STAT_METHODS)} "
            f"(Poisson likelihoods); got {resolved!r}. Pass stat_method= or set "
            f"the global FitConfig.statistic accordingly."
        )
    return resolved


def fit_prepared_bxa(
    prepared,
    *,
    outdir: str | Path,
    emin: float | None = None,
    emax: float | None = None,
    energy_ranges: Mapping[str, tuple[float, float]] | None = None,
    model_name: str = "tbabs*ztbabs*cflux*powerlaw",
    frozen_parameters: Mapping[str, float] | None = None,
    redshift: float = 0.0,
    redshift_absorbers: Sequence[float] | None = None,
    stat_method: str | None = None,
    abundance: str | None = None,
    cross_section: str | None = None,
    galactic_nh_1e22: float | None = None,
    freeze_galactic_nh: bool = True,
    intrinsic_nh_mode: Literal["free", "zero"] = "free",
    prior_specs: Mapping[str, BXAPriorSpec] | None = None,
    # --- BXAConfig contract (field names == keyword names) ---
    n_live_points: int | None = None,
    evidence_tolerance: float = 0.5,
    speed: str = "safe",
    resume: bool = False,
    Lepsilon: float = 0.1,
    frac_remain: float | None = None,
    calculate_flux_chain: bool = True,
    flux_erange: str = "2.0 10.0",
    # --- diagnostic plots (best-effort; degrade to a warning on failure) ---
    plot_fit: bool = False,
    plot_posterior: bool = False,
    # --- naming / labels ---
    srcname: str | None = None,
    instname: str | None = None,
) -> BXAFitResult:
    """Run a BXA / UltraNest nested-sampling fit on prepared spectrum/spectra.

    The XSPEC session is assembled by reusing ``jinwu.core.fit`` private
    helpers (see module docstring), so the model, absorption handling, group
    linking and best-fit reporting are identical to :func:`fit_prepared`.  On
    top of that session BXA builds prior transformations for the thawed,
    unlinked parameters and samples the posterior with ``vectorized=False``.

    ``stat_method`` must be a Poisson statistic (``cstat``/``cash``/``pstat``).
    The output directory and its ``chains`` sub-directory are created up front
    (BXA's ``outputfiles_basename`` must already exist).  The XSPEC global
    session is cleared on entry *and* exit.
    """
    prepared_spectra = _resolve_prepared_spectra(prepared)

    xspec = _require_xspec()
    bxa = _require_bxa()

    settings = get_fit_settings()
    resolved_stat = _resolve_stat_method(stat_method)
    abundance = abundance if abundance is not None else settings.abundance
    cross_section = (
        cross_section if cross_section is not None else settings.cross_section
    )

    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chains_dir = output / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []
    fit_ranges = _prepared_energy_ranges(
        prepared_spectra, emin=emin, emax=emax, energy_ranges=energy_ranges
    )
    first_key = _prepared_spectrum_key(prepared_spectra[0])
    fit_emin, fit_emax = fit_ranges[first_key]

    xspec.AllData.clear()
    xspec.AllModels.clear()
    try:
        xspec.Xset.abund = abundance
        xspec.Xset.xsect = cross_section
        xspec.Fit.query = "yes"
        xspec.Fit.statMethod = resolved_stat

        data_groups = _prepared_data_groups(prepared_spectra, fit_ranges)
        group_for_index = {
            spectrum_index: group
            for group in data_groups
            for spectrum_index in group["spectrum_indices"]
        }
        xspec_spectra = []
        if len(prepared_spectra) == 1:
            xspec_spectrum = _load_prepared_xspec_spectrum(xspec, prepared_spectra[0])
            _set_prepared_xspec_links(xspec_spectrum, prepared_spectra[0])
            xspec_spectra.append(xspec_spectrum)
        else:
            for index, spectrum in enumerate(prepared_spectra, start=1):
                xspec_spectrum = _load_prepared_xspec_spectrum(
                    xspec,
                    spectrum,
                    index=index,
                    data_group=group_for_index[index]["group_index"],
                )
                _set_prepared_xspec_links(xspec_spectrum, spectrum)
                xspec_spectra.append(xspec_spectrum)

        xspec.AllData.ignore("bad")
        for xspec_spectrum, spectrum in zip(xspec_spectra, prepared_spectra):
            spectrum_emin, spectrum_emax = fit_ranges[_prepared_spectrum_key(spectrum)]
            xspec_spectrum.ignore(f"**-{spectrum_emin} {spectrum_emax}-**")

        model = xspec.Model(model_name)
        group_models = []
        for group in data_groups:
            group_model = _xspec_model_for_group(xspec, model, group["group_index"])
            group_models.append(group_model)
            _configure_prepared_model(
                group_model,
                model_name=model_name,
                emin=group["energy_range"]["emin"],
                emax=group["energy_range"]["emax"],
                redshift=redshift,
                redshift_absorbers=redshift_absorbers,
                galactic_nh_1e22=galactic_nh_1e22,
                freeze_galactic_nh=freeze_galactic_nh,
                intrinsic_nh_mode=intrinsic_nh_mode,
            )
            _freeze_prepared_parameters(group_model, frozen_parameters)
        _link_default_prepared_model_groups(group_models, model_name)

        # Maximum-likelihood warm start so nested sampling begins near the peak.
        perform_text = _capture_xspec_log(
            xspec,
            output / "fit_prepared_bxa.xspec_perform.tmp.log",
            xspec.Fit.perform,
            warnings_list,
        )

        transformations = resolve_priors(xspec, model, group_models, prior_specs)
        paramnames = [str(item["name"]) for item in transformations]

        solver = bxa.BXASolver(
            transformations, outputfiles_basename=str(chains_dir)
        )
        # BXA's ``BXASolver.run`` accepts either a keyword speed string
        # ("safe"/"auto") or an ``int`` used as ``SliceSampler(nsteps=speed)``.
        # Only stringify the keyword form; pass integers through untouched so
        # the ultranest ``1.1 ** (1.0 / nsteps)`` arithmetic never sees a str.
        resolved_speed = speed if isinstance(speed, int) else str(speed)
        resolved_frac_remain = (
            None if frac_remain is None else float(frac_remain)
        )
        run_settings = {
            "speed": resolved_speed,
            "resume": bool(resume),
            "n_live_points": n_live_points,
            "evidence_tolerance": float(evidence_tolerance),
            "Lepsilon": float(Lepsilon),
            "frac_remain": resolved_frac_remain,
        }
        solver.run(
            speed=resolved_speed,
            resume=bool(resume),
            n_live_points=n_live_points,
            evidence_tolerance=float(evidence_tolerance),
            Lepsilon=float(Lepsilon),
            frac_remain=resolved_frac_remain,
        )
        # ``BXASolver.run`` already calls ``set_best_fit`` internally; repeat it
        # so the exported session reflects the maximum-likelihood point.
        solver.set_best_fit()

        results = solver.results
        logz = _finite_or_none(results.get("logz"))
        logzerr = _finite_or_none(results.get("logzerr"))
        posterior = results.get("samples")
        marginals = results.get("marginals")

        best_fit = _generate_xspec_result(
            model,
            xspec_spectra[0],
            flux_range_keV=fit_ranges[first_key],
            warnings_list=warnings_list,
            errors_computed=False,
        )

        flux_chain = None
        if calculate_flux_chain:
            try:
                flux_chain = solver.create_flux_chain(
                    xspec_spectra[0], erange=str(flux_erange)
                )
            except Exception as exc:  # pragma: no cover - depends on data
                warnings_list.append(f"BXA flux chain failed: {exc}")

        fit_statistic = _finite_or_none(getattr(xspec.Fit, "statistic", None))
        fit_dof = _int_or_none(getattr(xspec.Fit, "dof", None))
        free_parameter_count = _count_free_xspec_parameters(group_models)
        source_counts, background_counts = _xspec_spectrum_counts(
            xspec_spectra, warnings_list
        )

        if srcname is None:
            first = prepared_spectra[0]
            srcname = (
                "_".join(
                    str(value)
                    for value in (
                        first.obsid,
                        first.module or first.detector,
                        first.source_id,
                    )
                    if value
                )
                or "prepared"
            )
        if instname is None:
            instname = "+".join(
                spectrum.module or spectrum.detector or spectrum.instrument
                for spectrum in prepared_spectra
            )
        label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{srcname}_{instname}").strip("_")

        prepared_inputs = [
            _prepared_input_dict(spectrum) for spectrum in prepared_spectra
        ]
        show_text = _xspec_show_text(xspec, output, label, warnings_list)
        report = _write_prepared_report(
            path=output / f"{label}_bxa_fit.txt",
            prepared_inputs=prepared_inputs,
            results=best_fit,
            show_text=show_text,
            error_command=None,
            error_text="",
            warnings_list=warnings_list,
            data_groups=data_groups,
        )

        plot_paths, plot_warnings = _generate_bxa_diagnostics(
            output=output,
            label=label,
            srcname=srcname,
            instname=instname,
            group_min=getattr(prepared_spectra[0], "group_min", None),
            model_name=model_name,
            redshift=redshift,
            posterior=posterior,
            paramnames=paramnames,
            plot_fit=plot_fit,
            plot_posterior=plot_posterior,
        )
        warnings_list.extend(plot_warnings)

        chain_path = chains_dir / "chain.fits"
        from jinwu.core.products import save_xspec_session

        products = save_xspec_session(
            xspec,
            output_dir=output,
            label=f"{label}_bxa",
            result=best_fit,
            report_txt=report,
            transcript="\n\n".join(
                part
                for part in (perform_text.strip(), show_text.strip())
                if part
            ),
            plots=plot_paths,
            input_paths=[
                value
                for item in prepared_inputs
                for value in item.values()
                if isinstance(value, str) and Path(value).exists()
            ],
        )
        fit_products = {
            "result_json": str(products.result_json),
            "report_txt": str(products.report_txt),
            "xspec_log": str(products.xspec_log),
            "xcm": str(products.xcm),
            "replay_cwd": str(products.replay_cwd),
            "plots": [str(path) for path in products.plots],
        }

        summary_path = output / "bxa_summary.json"
        summary_payload = {
            "model_name": model_name,
            "paramnames": paramnames,
            "logz": logz,
            "logzerr": logzerr,
            "fit_statistic": fit_statistic,
            "fit_dof": fit_dof,
            "free_parameter_count": free_parameter_count,
            "stat_method": resolved_stat,
            "n_live_points": n_live_points,
            "evidence_tolerance": float(evidence_tolerance),
            "speed": resolved_speed,
            "Lepsilon": float(Lepsilon),
            "frac_remain": resolved_frac_remain,
            "resume": bool(resume),
            "chain_path": str(chain_path) if chain_path.exists() else None,
            "flux_erange": str(flux_erange) if calculate_flux_chain else None,
            "energy_range": {"emin": fit_emin, "emax": fit_emax},
            "run_settings": run_settings,
            "fit_products": fit_products,
            "warnings": warnings_list,
        }
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return BXAFitResult(
            model_name=model_name,
            paramnames=paramnames,
            logz=logz,
            logzerr=logzerr,
            posterior=posterior,
            marginals=marginals,
            best_fit=best_fit,
            fit_statistic=fit_statistic,
            fit_dof=fit_dof,
            free_parameter_count=free_parameter_count,
            source_counts=source_counts,
            background_counts=background_counts,
            energy_range={"emin": fit_emin, "emax": fit_emax},
            energy_ranges={
                key: {"emin": limits[0], "emax": limits[1]}
                for key, limits in fit_ranges.items()
            },
            chain_path=str(chain_path) if chain_path.exists() else None,
            flux_chain=flux_chain,
            n_live_points=n_live_points,
            stat_method=resolved_stat,
            run_settings=run_settings,
            parallel_settings={"vectorized": False, "backend": "ultranest"},
            cache_status="resume" if resume else "overwrite",
            warnings=warnings_list,
            status="ok",
            fit_products=fit_products,
        )
    finally:
        # PyXspec is a non-reentrant global singleton; always leave it clean.
        try:
            xspec.AllData.clear()
            xspec.AllModels.clear()
        except Exception:  # pragma: no cover - defensive
            pass


def _generate_bxa_diagnostics(
    *,
    output: Path,
    label: str,
    srcname: str,
    instname: str,
    group_min: int | None,
    model_name: str,
    redshift: float,
    posterior: Any,
    paramnames: Sequence[str],
    plot_fit: bool,
    plot_posterior: bool,
) -> tuple[list[str], list[str]]:
    """Best-effort diagnostic figures for a BXA fit.

    Returns ``(plot_paths, warnings)``.  Every failure -- a missing plotting
    dependency, an empty posterior, a backend error -- degrades to a warning so
    that requesting a diagnostic plot can never crash an otherwise successful
    nested-sampling run.  When both switches are off this is a cheap no-op.
    """
    plot_paths: list[str] = []
    warnings_out: list[str] = []

    if plot_fit:
        try:
            from jinwu.core.plot import plotfit

            figure_path, figure = plotfit(
                srcname=srcname,
                instname=instname,
                group_min=group_min if group_min is not None else 1,
                modelname=model_name,
                redshift=redshift,
                outputdir=output,
            )
            if figure_path is not None:
                plot_paths.append(str(figure_path))
            if figure is not None:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:  # pragma: no cover - defensive
                    pass
        except Exception as exc:  # pragma: no cover - depends on env/data
            warnings_out.append(f"BXA fit plot failed: {exc}")

    if plot_posterior:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            samples = (
                None if posterior is None else np.asarray(posterior, dtype=float)
            )
            if samples is None or samples.size == 0:
                raise ValueError("no posterior samples available to plot")
            samples = np.atleast_2d(samples)
            n_params = int(samples.shape[1])
            fig, axes = plt.subplots(
                n_params, 1, figsize=(6.0, 2.0 * n_params), squeeze=False
            )
            for index in range(n_params):
                axis = axes[index][0]
                axis.plot(samples[:, index], lw=0.8)
                name = (
                    paramnames[index]
                    if index < len(paramnames)
                    else f"param{index}"
                )
                axis.set_ylabel(str(name))
            axes[-1][0].set_xlabel("posterior sample")
            fig.tight_layout()
            posterior_path = output / f"{label}_bxa_posterior.png"
            fig.savefig(posterior_path, dpi=150)
            plt.close(fig)
            plot_paths.append(str(posterior_path))
        except Exception as exc:  # pragma: no cover - depends on env/data
            warnings_out.append(f"BXA posterior plot failed: {exc}")

    return plot_paths, warnings_out


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fit_xray_models_bxa(
    prepared,
    *,
    outdir: str | Path,
    model_class: str = "auto",
    absorption_mode: str = "auto",
    candidate_keys: Sequence[str] | None = None,
    galactic_nh_1e22: float | None = None,
    redshift: float = 0.0,
    prior_specs: Mapping[str, Mapping[str, BXAPriorSpec]] | None = None,
    **bxa_kwargs: Any,
) -> XRayModelComparisonResult:
    """Fit and compare X-ray model candidates by Bayesian evidence (logZ).

    Mirrors :func:`jinwu.core.fit.fit_xray_models` but each candidate is fitted
    with :func:`fit_prepared_bxa` and ranked by descending ``logz``.  The
    returned :class:`XRayModelComparisonResult` carries
    :class:`ModelFitMetrics` whose ``logz`` / ``logzerr`` are populated and
    whose ``ranking_metric`` is ``"logz"``.  ``prior_specs`` is keyed by
    candidate key, mapping to that candidate's ``{parameter: BXAPriorSpec}``.

    ``bxa_kwargs`` are forwarded to :func:`fit_prepared_bxa` (e.g.
    ``stat_method``, ``n_live_points``, ``evidence_tolerance``, ``speed``).
    """
    specs_tuple = resolve_xray_model_specs(
        model_class=model_class,
        absorption_mode=absorption_mode,
        candidate_keys=candidate_keys,
    )
    specs = {spec.key: spec for spec in specs_tuple}
    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not math.isfinite(float(redshift)) or float(redshift) < 0:
        raise ValueError("redshift must be finite and non-negative")
    if any("tbabs" in spec.expression.lower() for spec in specs_tuple):
        if galactic_nh_1e22 is None:
            raise ValueError(
                "galactic_nh_1e22 is required when any candidate contains TBabs"
            )
        if not math.isfinite(float(galactic_nh_1e22)) or float(galactic_nh_1e22) < 0:
            raise ValueError("galactic_nh_1e22 must be finite and non-negative")

    forbidden = {"model_name", "intrinsic_nh_mode", "outdir"}
    overlap = forbidden.intersection(bxa_kwargs)
    if overlap:
        raise TypeError(
            f"fit_xray_models_bxa controls these arguments: "
            f"{', '.join(sorted(overlap))}"
        )

    per_candidate_priors = dict(prior_specs or {})
    candidates: dict[str, dict[str, Any]] = {}
    raw_metrics: dict[str, ModelFitMetrics] = {}
    failures: dict[str, str] = {}
    warning_messages: list[str] = []

    for spec in specs_tuple:
        candidate_dir = output / "models" / spec.key
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = fit_prepared_bxa(
                prepared,
                outdir=candidate_dir,
                model_name=spec.expression,
                intrinsic_nh_mode=(
                    "zero" if spec.absorption_mode == "zero" else "free"
                ),
                galactic_nh_1e22=galactic_nh_1e22,
                freeze_galactic_nh=True,
                redshift=redshift,
                prior_specs=per_candidate_priors.get(spec.key),
                **bxa_kwargs,
            )
            if result.logz is None:
                raise RuntimeError(
                    f"BXA did not return a finite logZ for candidate {spec.key}"
                )
            metrics = calculate_bayesian_model_metrics(
                result.logz,
                result.logzerr,
                result.free_parameter_count,
                result.fit_dof if result.fit_dof is not None else 0,
            )
            payload = result.to_dict()
            payload["model_key"] = spec.key
            payload["model_family"] = spec.family
            payload["absorption_mode"] = spec.absorption_mode
            payload["metrics"] = {
                "statistic": metrics.statistic,
                "dof": metrics.dof,
                "free_parameters": metrics.free_parameters,
                "effective_bins": metrics.effective_bins,
                "aic": metrics.aic,
                "aicc": metrics.aicc,
                "bic": metrics.bic,
                "logz": metrics.logz,
                "logzerr": metrics.logzerr,
                "ranking_metric": metrics.ranking_metric,
            }
            candidates[spec.key] = payload
            raw_metrics[spec.key] = metrics
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failures[spec.key] = failure
            failure_payload = {
                "model_key": spec.key,
                "model_expression": spec.expression,
                "status": "failed",
                "error": failure,
            }
            (candidate_dir / "fit_failure.json").write_text(
                json.dumps(failure_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    if not candidates:
        raise RuntimeError(
            "All BXA model candidates failed: "
            + "; ".join(f"{key}: {value}" for key, value in failures.items())
        )

    ranking = tuple(
        sorted(candidates, key=lambda key: raw_metrics[key].logz, reverse=True)
    )
    best_logz = raw_metrics[ranking[0]].logz
    metrics: dict[str, ModelFitMetrics] = {}
    for key in ranking:
        base = raw_metrics[key]
        metrics[key] = replace(base, delta=(best_logz - base.logz))
    adopted_key = ranking[0]
    adopted_reason = (
        f"Highest Bayesian log-evidence (logZ={best_logz:.3f}) among "
        f"{len(candidates)} BXA candidate(s)."
    )

    comparison = XRayModelComparisonResult(
        candidates=candidates,
        metrics=metrics,
        failures=failures,
        ranking=ranking,
        adopted_key=adopted_key,
        adopted_reason=adopted_reason,
        selection_metric="logz",
        warnings=tuple(warning_messages),
    )
    json_path = output / "model_comparison.json"
    text_path = output / "model_comparison.txt"
    comparison.comparison_json = str(json_path)
    comparison.comparison_txt = str(text_path)

    comparison_payload = {
        "adopted_key": adopted_key,
        "adopted_reason": adopted_reason,
        "selection_metric": "logz",
        "ranking": list(ranking),
        "metrics": {
            key: {
                "logz": value.logz,
                "logzerr": value.logzerr,
                "dof": value.dof,
                "free_parameters": value.free_parameters,
                "delta": value.delta,
                "ranking_metric": value.ranking_metric,
            }
            for key, value in metrics.items()
        },
        "failures": dict(failures),
        "warnings": list(warning_messages),
    }
    json_path.write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text_lines = [
        f"Adopted model: {adopted_key}",
        f"Reason: {adopted_reason}",
        "Ranking metric: logz",
        "",
        "key logZ logZerr dof k delta",
    ]
    for key in ranking:
        item = metrics[key]
        logzerr = f"{item.logzerr:.4g}" if item.logzerr is not None else "N/A"
        text_lines.append(
            f"{key} {item.logz:.4f} {logzerr} {item.dof} "
            f"{item.free_parameters} {item.delta:.4f}"
        )
    if failures:
        text_lines.extend(["", "Failed candidates:"])
        text_lines.extend(f"{key}: {value}" for key, value in failures.items())
    if warning_messages:
        text_lines.extend(["", "Warnings:", *warning_messages])
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return comparison


def _build_jinwu_model_solver(base_cls):
    """Construct the ``JinwuModelBXASolver`` subclass of ``base_cls``.

    The subclass is built lazily (never at import time) so ``jinwu.core`` stays
    importable without BXA.  Its ``log_likelihood`` is a placeholder that must
    be implemented once a native ``jinwu.model`` likelihood engine exists; for
    now it raises ``NotImplementedError``.
    """

    class JinwuModelBXASolver(base_cls):  # type: ignore[misc, valid-type]
        """BXA solver backed by a native ``jinwu.model`` likelihood (stub).

        Reserved for the future non-XSPEC likelihood engine.  The current
        XSPEC-backed path (:func:`fit_prepared_bxa`) is unaffected.
        """

        def log_likelihood(self, params):  # pragma: no cover - stub
            raise NotImplementedError(
                "JinwuModelBXASolver.log_likelihood requires a native "
                "jinwu.model likelihood engine, which is not implemented yet. "
                "Use fit_prepared_bxa (XSPEC-backed BXA) for now."
            )

    JinwuModelBXASolver.__name__ = "JinwuModelBXASolver"
    JinwuModelBXASolver.__qualname__ = "JinwuModelBXASolver"
    return JinwuModelBXASolver


def fit_prepared_jinwu_model(*args: Any, **kwargs: Any):
    """Placeholder for a native ``jinwu.model`` Bayesian fit (not exported).

    Raises ``NotImplementedError``: reserved for the future non-XSPEC
    likelihood backend.  Deliberately excluded from ``__all__``.
    """
    raise NotImplementedError(
        "fit_prepared_jinwu_model is a placeholder for the future native "
        "jinwu.model likelihood backend; use fit_prepared_bxa (XSPEC-backed "
        "BXA nested sampling) instead."
    )


def run_bxa_pipeline(
    data,
    *,
    outdir: str | Path,
    group_min: int | None = None,
    prepare_outdir: str | Path | None = None,
    model_name: str = "tbabs*ztbabs*cflux*powerlaw",
    overwrite: bool = False,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate prepare -> BXA fit -> aggregate for one scanned dataset.

    ``data`` is a :class:`~jinwu.core.spectrum_prep.Catalog` /
    ``Manifest``.  Each ready prepared spectrum is fitted with
    :func:`fit_prepared_bxa`; the returned ``dict`` collects per-spectrum
    ``logz`` / ``logzerr`` and product paths plus any failures.  ``backend_kwargs``
    are forwarded to :func:`fit_prepared_bxa` (including ``BXAConfig`` fields).
    """
    from jinwu.core.spectrum_prep import prepare_spectra

    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prep_root = (
        Path(prepare_outdir).expanduser().resolve()
        if prepare_outdir is not None
        else output / "prepared"
    )
    catalog = prepare_spectra(
        data, outdir=prep_root, group_min=group_min, overwrite=overwrite
    )

    kwargs = dict(backend_kwargs or {})
    fits: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for spectrum in catalog.spectra:
        if not spectrum.ready or spectrum.grouped_pha is None:
            key = _prepared_spectrum_key(spectrum)
            failures[key] = f"not fit-ready (status={spectrum.status})"
            continue
        key = _prepared_spectrum_key(spectrum)
        label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", key).strip("_") or "spectrum"
        fit_dir = output / "fits" / label
        try:
            result = fit_prepared_bxa(
                spectrum, outdir=fit_dir, model_name=model_name, **kwargs
            )
            fits.append(
                {
                    "key": key,
                    "outdir": str(fit_dir),
                    "logz": result.logz,
                    "logzerr": result.logzerr,
                    "fit_statistic": result.fit_statistic,
                    "fit_dof": result.fit_dof,
                    "chain_path": result.chain_path,
                    "fit_products": result.fit_products,
                }
            )
        except Exception as exc:
            failures[key] = f"{type(exc).__name__}: {exc}"

    summary = {
        "outdir": str(output),
        "prepared_root": str(prep_root),
        "prepared_status": catalog.status,
        "model_name": model_name,
        "n_fits": len(fits),
        "fits": fits,
        "failures": failures,
        "warnings": list(catalog.warnings),
    }
    (output / "bxa_pipeline.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def __getattr__(name: str) -> Any:
    """Lazily expose ``JinwuModelBXASolver`` without importing BXA at module
    load time.  Constructing the subclass requires ``bxa``; accessing it in a
    dependency-free environment raises the standard install guidance.
    """
    if name == "JinwuModelBXASolver":
        bxa = _require_bxa()
        cls = _build_jinwu_model_solver(bxa.BXASolver)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
