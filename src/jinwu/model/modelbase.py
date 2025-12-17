"""
Model base classes inspired by XSPEC:
 AdditiveModel: produces a spectrum to be summed
 MultiplicativeModel: produces a transmission factor to multiply
 ConvolutionModel: convolves an input spectrum
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np


class ModelBase(ABC):
    """Common base class for all spectral model components."""

    def __init__(self, name: Optional[str] = None, params: Optional[Dict[str, float]] = None):
        self.name = name or self.__class__.__name__
        self.params: Dict[str, float] = params.copy() if params else {}

    @property
    def param_names(self) -> Iterable[str]:
        return self.params.keys()

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self.params:
                raise KeyError(f"Unknown parameter '{k}' for model '{self.name}'")
            self.params[k] = float(v)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Return model output. Subclasses define the signature."""
        raise NotImplementedError


class AdditiveModel(ModelBase):
    """Additive component: returns a spectrum to be summed."""

    @abstractmethod
    def evaluate(self, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        energy : np.ndarray
            Energy array (bin edges or centers).
        Returns
        -------
        np.ndarray
            Model spectrum in the same shape as energy input (or len-1 for bin edges).
        """
        raise NotImplementedError


class MultiplicativeModel(ModelBase):
    """Multiplicative component: returns a transmission factor."""

    @abstractmethod
    def evaluate(self, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        energy : np.ndarray
            Energy array (bin edges or centers).
        Returns
        -------
        np.ndarray
            Multiplicative factor matching the spectrum shape.
        """
        raise NotImplementedError

    def apply(self, energy: np.ndarray, spectrum: np.ndarray, **kwargs) -> np.ndarray:
        """Apply multiplicative factor to a given spectrum."""
        factor = self.evaluate(energy, **kwargs)
        return np.asarray(spectrum) * np.asarray(factor)


class ConvolutionModel(ModelBase):
    """Convolution component: transforms an input spectrum."""

    @abstractmethod
    def evaluate(self, energy: np.ndarray, spectrum: np.ndarray, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        energy : np.ndarray
            Energy array (bin edges or centers).
        spectrum : np.ndarray
            Input spectrum to be convolved.
        Returns
        -------
        np.ndarray
            Convolved spectrum, same shape as input spectrum.
        """
        raise NotImplementedError

    def apply(self, energy: np.ndarray, spectrum: np.ndarray, **kwargs) -> np.ndarray:
        """Alias for evaluate: perform convolution on the input spectrum."""
        return self.evaluate(energy, spectrum, **kwargs)
