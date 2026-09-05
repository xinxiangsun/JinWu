# BAT submodule for jinwu
# BAT 子模块

from .bat_observation import BATObservation
from .attitude import Attitude

# Keep the optional BatAnalysis adapter lazy.  Besides avoiding an optional
# dependency at package import time, this prevents ``python -m
# jinwu.swift.bat.survey`` from importing the module once through the package
# and then executing it a second time under ``__main__``.
_SURVEY_EXPORTS = (
    'BATSurveyInput', 'BATSurveyPipeline', 'BATSurveyResult',
    'BatAnalysisSurveyBackend', 'AreaRow', 'SurveyRatePoint',
    'SurveyPhaValidation', 'SurveyUpperLimitResult', 'build_source_catalog',
    'calculate_background_scale', 'read_gti_intervals',
    'gti_overlap_duration', 'parse_area_table', 'read_bat_survey_rates',
    'safe_extract_archive', 'select_overlapping_pointings', 'signed_snr',
    'validate_observation_directory', 'validate_survey_pha',
    'profile_survey_upper_limit', 'estimate_bat_survey_sensitivity',
    'BATSurveySensitivityAdapter',
)

__all__ = [
    'BATObservation', 'Attitude',
    *_SURVEY_EXPORTS,
]


def __getattr__(name):
    if name in _SURVEY_EXPORTS:
        from . import survey
        value = getattr(survey, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
