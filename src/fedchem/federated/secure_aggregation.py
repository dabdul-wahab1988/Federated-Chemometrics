from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np

class SecureAggregator:
    """Abstract interface for secure aggregation adapters.

    Implementations may use cryptographic secure aggregation libraries or
    provide a simulated/functional test harness used during development.
    """

    def prepare_client_masked_update(self, client_id: str, update: np.ndarray) -> np.ndarray:
        """Return a masked update for the given client that when aggregated with
        other masked updates and post-processed yields the true sum.

        Args:
            client_id: opaque client identifier (used for RNG seed in simulated)
            update: raw update vector (np.ndarray)
        Returns:
            masked_update: np.ndarray masked by a random vector for privacy simulation.
        """
        raise NotImplementedError()

    def aggregate_masked_updates(self, masked_updates: List[np.ndarray]) -> np.ndarray:
        """Given masked updates, return aggregated sum (masked aggregation)."""
        raise NotImplementedError()

    def unmask_aggregate(self, agg: np.ndarray) -> np.ndarray:
        """Return the unmasked aggregated update (server side)."""
        raise NotImplementedError()


class SimulatedSecureAggregator(SecureAggregator):
    """A deterministic, simulated secure aggregator for testing.

    The simplest possible secure aggregation is implemented by generating
    zero-sum random masks across participating clients so that the server can
    recover the true sum while intermediate masked payloads don't reveal
    individual updates.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)
        self._client_masks: Dict[str, np.ndarray] = {}
        self._mask_dim: int | None = None

    def _ensure_mask(self, client_id: str, dim: int) -> None:
        if self._mask_dim is None:
            self._mask_dim = int(dim)
        if client_id in self._client_masks:
            return
        # For simulation, sample random masks; the server will sum them and
        # subtract a final mask to ensure zero-sum across clients when all are
        # known to the server. For simplicity, use a fixed RNG per-client id
        # (deterministic) so that tests are reproducible.
        mask = self.rng.normal(0.0, 1.0, size=(dim,))
        self._client_masks[client_id] = mask

    def prepare_client_masked_update(self, client_id: str, update: np.ndarray) -> np.ndarray:
        update_vec = np.asarray(update).ravel()
        dim = update_vec.size
        self._ensure_mask(client_id, dim)
        mask = self._client_masks[client_id]
        if mask.shape[0] != dim:
            # if dimension mismatch, reinitialize mask for this client
            mask = self.rng.normal(0.0, 1.0, size=(dim,))
            self._client_masks[client_id] = mask
        masked = update_vec + mask
        return masked

    def aggregate_masked_updates(self, masked_updates: List[np.ndarray]) -> np.ndarray:
        # simple sum across masked updates
        total = np.sum(np.vstack([m.ravel() for m in masked_updates]), axis=0)
        return total

    def unmask_aggregate(self, agg: np.ndarray) -> np.ndarray:
        # subtract the sum of masks
        masks = list(self._client_masks.values())
        if not masks:
            return agg
        mask_sum = np.sum(np.vstack([m.ravel() for m in masks]), axis=0)
        return agg - mask_sum
