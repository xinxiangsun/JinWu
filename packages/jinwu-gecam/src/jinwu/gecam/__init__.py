"""GECAMTools interfaces; the optional ``gecam`` backend loads on use only."""
from importlib.metadata import PackageNotFoundError, version

from .analysis import prepare_gecam_spectra, gecam_lightcurve_products
from .backend import extract_lightcurve, extract_spectra, open_event
from .pipeline import GECAMConfig, GECAMInput, GECAMPipeline

__all__ = ["prepare_gecam_spectra", "gecam_lightcurve_products", "GECAMConfig", "GECAMInput", "GECAMPipeline", "open_event", "extract_lightcurve", "extract_spectra"]

try:
    __version__ = version("jinwu-gecam")
except PackageNotFoundError:
    __version__ = "0.0.0"
