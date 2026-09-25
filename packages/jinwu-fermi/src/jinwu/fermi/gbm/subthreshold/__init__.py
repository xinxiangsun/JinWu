"""GBM external-trigger subthreshold searches (optional ``search`` extra).

The search reuses the attributed USRA GTS likelihood. It returns candidates;
statistical significance requires a matching empirical off-source calibration.
"""
from .models import GBMTargetedSearchInput, GBMTargetedSearchConfig, GBMTargetedSearchResult
from .pipeline import GBMTargetedSearchPipeline, run_targeted_search
from .calibration import calibrate_targeted_search, calibration_from_searches, estimate_candidate_far

__all__ = ["GBMTargetedSearchInput", "GBMTargetedSearchConfig", "GBMTargetedSearchResult",
           "GBMTargetedSearchPipeline", "run_targeted_search", "calibrate_targeted_search",
           "calibration_from_searches", "estimate_candidate_far"]
