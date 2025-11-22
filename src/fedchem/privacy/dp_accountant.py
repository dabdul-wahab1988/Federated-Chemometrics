from __future__ import annotations

from typing import Dict, Iterable, Optional
import math


class DPAccountant:
    """Differential privacy accounting utilities.

    Implements Renyi DP (RDP) composition for the Gaussian mechanism and
    simple composition for the Laplace mechanism. This provides research-grade
    accounting suitable for configuring FL runs with clipping and noise.

    Conventions:
    - Gaussian mechanism parameterized by standard deviation `sigma` and L2
      sensitivity `sensitivity` (default 1.0). Per-round RDP at order α is
      α * sensitivity^2 / (2 * sigma^2). Composition across rounds adds RDP.
      Convert to (ε, δ) via ε = min_α RDP(α) + log(1/δ)/(α-1).
    - Laplace mechanism uses pure DP with scale b. If `epsilon_per_round` is
      specified, ε_total = rounds * epsilon_per_round. If only `b` and
      `sensitivity` provided, epsilon_per_round = sensitivity / b.
    """

    def _rdp_gaussian(self, alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
        if alpha <= 1:
            raise ValueError("RDP order alpha must be > 1")
        return alpha * (sensitivity ** 2) / (2.0 * (sigma ** 2))

    def _orders(self) -> Iterable[float]:
        # Reasonable sweep of orders for minimization, include some fractional > 2
        return [
            1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 8.0, 10.0, 16.0, 32.0, 64.0, 128.0, 256.0
        ]

    @staticmethod
    def _log_comb(n: int, k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def _rdp_gaussian_subsampled_int(self, alpha: int, q: float, sigma: float) -> float:
        """Exact RDP for Poisson-subsampled Gaussian (integer alpha >= 2).

        Implements the binomial-mixture expression (Wang et al., 2019):
        A_α = sum_{k=0..α} C(α, k) q^k (1-q)^{α-k} exp(k(k-1)/(2σ^2))
        RDP(α) = (1/(α-1)) * log A_α
        """
        if alpha < 2:
            raise ValueError("alpha must be integer >= 2 for subsampled bound")
        inv_sigma2 = 1.0 / (sigma * sigma)
        logs = []
        logq = math.log(max(1e-20, q))
        log1q = math.log(max(1e-20, 1.0 - q))
        for k in range(0, alpha + 1):
            term_log = (
                self._log_comb(alpha, k)
                + k * logq
                + (alpha - k) * log1q
                + 0.5 * k * (k - 1) * inv_sigma2
            )
            logs.append(term_log)
        m = max(logs)
        A = m + math.log(sum(math.exp(L - m) for L in logs))
        return A / (alpha - 1)

    def _rdp_gaussian_subsampled(self, alpha: float, q: float, sigma: float) -> float:
        """RDP for Poisson-subsampled Gaussian for real alpha >= 2 via interpolation.

        Uses exact integer orders when possible; for fractional orders, returns
        the convex interpolation between floor and ceil integers (>=2).
        For alpha < 2, falls back to alpha=2 as a safe upper bound.
        """
        if alpha < 2:
            return self._rdp_gaussian_subsampled_int(2, q, sigma)
        if float(alpha).is_integer():
            return self._rdp_gaussian_subsampled_int(int(alpha), q, sigma)
        a0 = max(2, int(math.floor(alpha)))
        a1 = a0 + 1
        r0 = self._rdp_gaussian_subsampled_int(a0, q, sigma)
        r1 = self._rdp_gaussian_subsampled_int(a1, q, sigma)
        t = alpha - a0
        return (1.0 - t) * r0 + t * r1

    def compute(
        self,
        mech: str,
        params: dict,
        rounds: int,
        *,
        sensitivity: Optional[float] = None,
    ) -> Dict[str, float]:
        if rounds <= 0:
            raise ValueError("rounds must be > 0")
        mech = mech.lower()
        if mech not in {"gaussian", "laplace", "gaussian_subsampled"}:
            raise ValueError("Unsupported mechanism")

        if mech == "gaussian":
            sigma = float(params.get("sigma", 1.0))
            delta = float(params.get("delta", 1e-5))
            sens = float(1.0 if sensitivity is None else sensitivity)
            best_eps = math.inf
            best_alpha = 2.0
            for a in self._orders():
                rdp = rounds * self._rdp_gaussian(a, sigma, sensitivity=sens)
                eps = rdp + math.log(1.0 / max(1e-20, delta)) / (a - 1.0)
                if eps < best_eps:
                    best_eps = eps
                    best_alpha = a
            return {"epsilon": float(best_eps), "delta": float(delta), "order": float(best_alpha)}

        if mech == "gaussian_subsampled":
            sigma = float(params.get("sigma", 1.0))
            delta = float(params.get("delta", 1e-5))
            q = float(params.get("q", 0.1))
            sens = float(1.0 if sensitivity is None else sensitivity)
            # Scale sigma by sensitivity
            sigma_eff = sigma / max(1e-20, sens)
            best_eps = math.inf
            best_alpha = 2.0
            # Use mixed orders including fractional via interpolation
            for a in self._orders():
                if a < 2:
                    continue
                rdp = rounds * (self._rdp_gaussian_subsampled(a, q, sigma_eff))
                eps = rdp + math.log(1.0 / max(1e-20, delta)) / (a - 1.0)
                if eps < best_eps:
                    best_eps = eps
                    best_alpha = a
            return {"epsilon": float(best_eps), "delta": float(delta), "order": float(best_alpha)}

        # Laplace
        delta = float(params.get("delta", 0.0))
        eps_round = params.get("epsilon_per_round")
        if eps_round is None:
            b = float(params.get("b", 1.0))
            sens = float(1.0 if sensitivity is None else sensitivity)
            eps_round = sens / max(1e-12, b)
        epsilon = rounds * float(eps_round)
        return {"epsilon": float(epsilon), "delta": float(delta)}

    def solve_sigma_for_target_epsilon(
        self,
        target_epsilon: float,
        delta: float,
        rounds: int,
        *,
        sensitivity: float = 1.0,
        q: Optional[float] = None,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        tol: float = 1e-4,
        max_iter: int = 60,
    ) -> float:
        """Find Gaussian noise sigma achieving target epsilon at given delta.

        Uses binary search over sigma, minimizing computed epsilon from RDP.
        """
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be > 0")
        lo, hi = sigma_min, sigma_max
        for _ in range(max_iter):
            mid = math.sqrt(lo * hi)
            mech = "gaussian_subsampled" if (q is not None and q > 0 and q < 1) else "gaussian"
            params = {"sigma": mid, "delta": delta}
            if mech == "gaussian_subsampled":
                assert q is not None
                params["q"] = float(q)
            eps = self.compute(mech, params, rounds, sensitivity=sensitivity)["epsilon"]
            if eps > target_epsilon:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) / max(1.0, lo) < tol:
                break
        return hi

    def compute_progression(
        self,
        mech: str,
        params_list: list[dict],
        delta: float,
        *,
        sensitivity: float = 1.0,
    ) -> list[dict]:
        """Compute cumulative epsilon progression for a sequence of mechanism parameters.

        `params_list` is a list where each entry contains mechanism-specific params per round.
        For `gaussian_subsampled`, each entry must contain `sigma` and `q`.
        Returns a list of dicts with keys `epsilon` and `order` for each prefix [1..N].
        """
        mech = mech.lower()
        if mech not in {"gaussian", "laplace", "gaussian_subsampled"}:
            raise ValueError("Unsupported mechanism")
        n = len(params_list)
        if n == 0:
            return []
        out: list[dict] = []
        # For gaussian_subsampled, we sum RDP contributions per alpha
        if mech == "gaussian_subsampled" or mech == "gaussian":
            # Prepare per-round rdp(alpha) contributions for each alpha
            orders = list(self._orders())
            # Precompute per-round RDPs: rdp_per_round[i][j] for round i and order j
            rdp_per_round: list[list[float]] = []
            for p in params_list:
                sigma = float(p.get("sigma", 1.0))
                q = float(p.get("q", 1.0 if mech == "gaussian" else p.get("q", 1.0)))
                # Convert sensitivity scaling
                sigma_eff = sigma / max(1e-20, float(sensitivity))
                row: list[float] = []
                for a in orders:
                    if mech == "gaussian":
                        row.append(self._rdp_gaussian(a, sigma_eff, sensitivity=sensitivity))
                    else:
                        if a < 2:
                            row.append(self._rdp_gaussian_subsampled_int(2, q, sigma_eff))
                        elif float(a).is_integer():
                            row.append(self._rdp_gaussian_subsampled_int(int(a), q, sigma_eff))
                        else:
                            row.append(self._rdp_gaussian_subsampled(a, q, sigma_eff))
                rdp_per_round.append(row)

            # Compute prefix cumulative RDP and convert to epsilon at delta using best alpha
            for k in range(1, n + 1):
                cum_rdp = [0.0 for _ in orders]
                for i in range(k):
                    for j, val in enumerate(rdp_per_round[i]):
                        cum_rdp[j] += val
                best_eps = math.inf
                best_alpha = orders[0]
                for j, a in enumerate(orders):
                    rdp = cum_rdp[j]
                    eps = rdp + math.log(1.0 / max(1e-20, float(delta))) / (a - 1.0)
                    if eps < best_eps:
                        best_eps = eps
                        best_alpha = a
                out.append({"epsilon": float(best_eps), "delta": float(delta), "order": float(best_alpha)})
            return out
        # For Laplace: params_list entries may have epsilon_per_round or b
        if mech == "laplace":
            cum_eps = 0.0
            for p in params_list:
                eps_round = p.get("epsilon_per_round")
                if eps_round is None:
                    b = float(p.get("b", 1.0))
                    eps_round = float(sensitivity) / max(1e-12, b)
                cum_eps += float(eps_round)
                out.append({"epsilon": float(cum_eps), "delta": float(delta)})
            return out
