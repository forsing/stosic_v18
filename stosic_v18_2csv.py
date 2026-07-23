from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v18_2csv.py — 7-node krug (K=7 / prilagodjenje 7/39) — Lie generator / T10 coherence iff (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/lie_generator_structure.pvs
  — thm_coherence_iff (T10):
      continuously_observed ∧ ¬shape_direction_reversal  IFF  coherent
  — shape_direction_reversal: cos(u_t, u_{t+1}) < 0 na obliku
  — u = Δ√p (tangent na Fisher sferi)

Mapiranje na 7/39:
  prozor W=7: μ_t = empirija draws[t−W:t]; u_t = Δ√μ
  reversal := cos(u_t,u_{t+1})<0; observed := True
  T10 coherent ⇒ skor += max(u_{t+1},0); next = top 7
  bez randoma; stop ako uzastopni/AP
"""

from typing import List

import numpy as np

from stosic_v1_2csv import CSV_LOTO, CSV_PLUS, EPS, MAX_NUM, load_draws
from stosic_v2_2csv import top7_from_freq
from stosic_v10_2csv import is_degenerate

COS_STRONG = 0.85
WIN = 7


def window_prob(draws: np.ndarray, start: int, end: int) -> np.ndarray:
    mu = np.zeros(MAX_NUM, dtype=np.float64)
    for d in draws[start:end]:
        for n in d:
            mu[int(n) - 1] += 1.0
    s = mu.sum()
    return mu / s if s > 0 else np.full(MAX_NUM, 1.0 / MAX_NUM)


def predict_next(draws: np.ndarray) -> List[int]:
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    # trebaju 3 uzastopna prozora → kraj bar WIN+2
    for t in range(WIN, len(draws) - 1):
        mu0 = window_prob(draws, t - WIN, t)
        mu1 = window_prob(draws, t - WIN + 1, t + 1)
        mu2 = window_prob(draws, t - WIN + 2, t + 2) if t + 2 <= len(draws) else None
        if mu2 is None:
            break
        s0 = np.sqrt(np.clip(mu0, EPS, None))
        s1 = np.sqrt(np.clip(mu1, EPS, None))
        s2 = np.sqrt(np.clip(mu2, EPS, None))
        u0 = s1 - s0
        u1 = s2 - s1
        n0 = float(np.linalg.norm(u0))
        n1 = float(np.linalg.norm(u1))
        if n0 < EPS or n1 < EPS:
            continue
        cos = float(np.dot(u0, u1) / (n0 * n1))
        continuously_observed = True
        shape_direction_reversal = cos < 0.0
        coherent = continuously_observed and (not shape_direction_reversal)
        if not coherent:
            continue
        w = 1.0 + max(cos, 0.0) + (1.0 if cos > COS_STRONG else 0.0)
        skor += w * np.maximum(u1, 0.0)

    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0

    combo = top7_from_freq(skor)
    if is_degenerate(combo):
        nu = np.zeros(MAX_NUM, dtype=np.float64)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    next_loto = predict_next(load_draws(CSV_LOTO))
    next_loto_plus = predict_next(load_draws(CSV_PLUS))
    if is_degenerate(next_loto):
        raise SystemExit("degenerisan next_loto (uzastopni/AP) — zaustavljen pre ispisa")
    if is_degenerate(next_loto_plus):
        raise SystemExit("degenerisan next_loto_plus (uzastopni/AP) — zaustavljen pre ispisa")
    print("next_loto:      ", next_loto)
    print("next_loto_plus: ", next_loto_plus)


if __name__ == "__main__":
    main()


"""
next_loto:       [13, x, 26, y, 30, z, 38]
next_loto_plus:  [8, x, 14, y, 23, z, 37]
"""



"""
v18: lie_generator_structure (T10) — koherentan generator na prozoru W=10.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
