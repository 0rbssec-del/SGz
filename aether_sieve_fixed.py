#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet : AETHER-SIEVE v1.0
(Anomalous Entropy, Topology & Harmonic Extraction Engine)
Pipeline unifié NumPy / Numba (@njit) / SciPy.fft pour la détection de forçage
informationnel, l'isolation de porteuses artificielles et l'analyse d'attracteurs.
"""

import time
import numpy as np
from numba import njit, prange
from scipy.fft import rfft, irfft, rfftfreq


# ============================================================================
# MODULE 1 : NOYAUX SPECTRAUX NUMBA (BLANCHIMENT ADAPTATIF & KURTOSIS)
# ============================================================================

@njit(fastmath=True, cache=True)
def _spectral_moments(psd: np.ndarray) -> tuple:
    """Calcule l'entropie spectrale normalisée [0, 1] et le Kurtosis spectral."""
    n = psd.shape[0]
    total_power = np.sum(psd)
    if total_power < 1e-15 or n <= 1:
        return 0.0, 0.0

    inv_power = 1.0 / total_power
    shannon = 0.0
    mean_p = total_power / float(n)
    m2 = 0.0
    m4 = 0.0

    for i in range(n):
        p = psd[i] * inv_power
        if p > 1e-15:
            shannon -= p * np.log(p)
        diff = psd[i] - mean_p
        d2 = diff * diff
        m2 += d2
        m4 += d2 * d2

    spec_entropy = shannon / np.log(float(n))
    var = m2 / float(n)
    spec_kurtosis = ((m4 / float(n)) / (var * var) - 3.0) if var > 1e-15 else 0.0
    return spec_entropy, spec_kurtosis


@njit(fastmath=True, cache=True)
def _adaptive_notch_gains(
    psd: np.ndarray,
    median_half_width: int,
    sigma_threshold: float
) -> tuple:
    """
    Détecte les pics de résonance artificiels au-dessus du plancher local 1/f
    via médiane glissante en log-espace et écart absolu médian (MAD).
    """
    n = psd.shape[0]
    log_psd = np.log(psd + 1e-15)
    baseline = np.empty(n, dtype=np.float64)

    baseline[0] = log_psd[0]
    for i in range(1, n):
        start = max(1, i - median_half_width)
        end = min(n, i + median_half_width + 1)
        baseline[i] = np.median(log_psd[start:end])

    residuals = log_psd - baseline
    mad = np.median(np.abs(residuals[1:])) + 1e-12
    robust_sigma = 1.4826 * mad

    gain_mask = np.ones(n, dtype=np.float64)
    anomaly_flags = np.zeros(n, dtype=np.uint8)

    total_energy = np.sum(psd[1:]) + 1e-15
    pumped_energy = 0.0

    for i in range(1, n):
        if (residuals[i] / robust_sigma) > sigma_threshold:
            anomaly_flags[i] = 1
            target_p = np.exp(baseline[i])
            gain_mask[i] = np.sqrt(target_p / (psd[i] + 1e-15))
            pumped_energy += max(0.0, psd[i] - target_p)

    return gain_mask, anomaly_flags, (pumped_energy / total_energy)


# ============================================================================
# MODULE 2 : NOYAUX TOPOLOGIQUES & ENTROPIQUES NUMBA (TSALLIS & TAKENS)
# ============================================================================

@njit(fastmath=True, cache=True)
def _tsallis_entropy_norm(window: np.ndarray, q: float, n_bins: int) -> float:
    """Entropie non-extensive de Tsallis S_q normalisée sur [0, 1]."""
    n = window.shape[0]
    w_min = np.min(window)
    w_max = np.max(window)
    if n <= 1 or (w_max - w_min) < 1e-12:
        return 0.0

    counts = np.zeros(n_bins, dtype=np.float64)
    scale = (n_bins - 1e-9) / (w_max - w_min)
    for i in range(n):
        counts[int((window[i] - w_min) * scale)] += 1.0

    inv_n = 1.0 / float(n)
    if abs(q - 1.0) < 1e-5:
        s = 0.0
        for b in range(n_bins):
            if counts[b] > 0.0:
                p = counts[b] * inv_n
                s -= p * np.log(p)
        return s / np.log(float(n_bins))

    sum_pq = 0.0
    for b in range(n_bins):
        if counts[b] > 0.0:
            sum_pq += (counts[b] * inv_n) ** q

    s_q = (1.0 - sum_pq) / (q - 1.0)
    s_q_max = (1.0 - (float(n_bins) ** (1.0 - q))) / (q - 1.0)
    return s_q / s_q_max if s_q_max > 0.0 else 0.0


@njit(fastmath=True, parallel=True, cache=True)
def _grassberger_procaccia_d2(
    series: np.ndarray,
    dim: int,
    tau: int,
    r_low: float,
    r_high: float
) -> float:
    """
    Projection directe de Takens et calcul parallélisé de la dimension
    de corrélation D2 sans allouer de matrice intermédiaire en mémoire.
    """
    n = series.shape[0]
    m = n - (dim - 1) * tau
    if m < 12:
        return 0.0

    r_low_sq = r_low * r_low
    r_high_sq = r_high * r_high
    c_low = 0
    c_high = 0

    for i in prange(m):
        loc_low = 0
        loc_high = 0
        for j in range(i + 1, m):
            dist_sq = 0.0
            for d in range(dim):
                diff = series[i + d * tau] - series[j + d * tau]
                dist_sq += diff * diff
            # Correctif : les deux `if` imbriqués dans prange font planter Numba
            # (0.61 et 0.68 : "unexpected cycle in lookup()"). Équivalent car r_low < r_high.
            loc_high += dist_sq < r_high_sq
            loc_low += dist_sq < r_low_sq
        c_low += loc_low
        c_high += loc_high

    if c_low == 0 or c_high <= c_low:
        return 0.0

    return np.log(float(c_high) / float(c_low)) / np.log(r_high / r_low)


@njit(fastmath=True, cache=True)
def _sliding_manifold_telemetry(
    stream: np.ndarray,
    window_size: int,
    step: int,
    q: float,
    n_bins: int,
    dim: int,
    tau: int
) -> tuple:
    """Exécute la télémétrie Tsallis + Takens + CER sur fenêtre glissante."""
    n = stream.shape[0]
    n_win = (n - window_size) // step + 1

    sq_arr = np.zeros(n_win, dtype=np.float64)
    dsq_arr = np.zeros(n_win, dtype=np.float64)
    d2_arr = np.zeros(n_win, dtype=np.float64)
    cer_arr = np.zeros(n_win, dtype=np.float64)

    for w in range(n_win):
        start = w * step
        win = stream[start : start + window_size]

        std = np.std(win)
        norm_w = (win - np.mean(win)) / std if std > 1e-12 else (win - np.mean(win))

        sq = _tsallis_entropy_norm(norm_w, q, n_bins)
        sq_arr[w] = sq
        dsq_arr[w] = (sq - sq_arr[w - 1]) if w > 0 else 0.0

        d2 = _grassberger_procaccia_d2(norm_w, dim, tau, 0.35, 1.15)
        d2_arr[w] = d2

        # Ratio d'Émergence Critique (équilibre organisation topologique / entropie)
        topo_lock = max(0.0, min(1.0, 1.0 - (d2 / float(dim))))
        cer_arr[w] = 0.65 * sq + 0.35 * topo_lock

    return sq_arr, dsq_arr, d2_arr, cer_arr


# ============================================================================
# MODULE 3 : ORCHESTRATEUR UNIFIÉ "AETHER-SIEVE"
# ============================================================================

class AetherSieveEngine:
    """
    Moteur d'isolation de signal et de profilage entropique/topologique.
    """
    def __init__(
        self,
        fs: float = 100.0,
        window_size: int = 512,
        step: int = 128,
        tsallis_q: float = 1.45,
        n_bins: int = 32,
        embed_dim: int = 4,
        tau: int = 6,
        median_half_width: int = 25,
        sigma_threshold: float = 3.8
    ):
        self.fs = fs
        self.window_size = window_size
        self.step = step
        self.tsallis_q = tsallis_q
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        self.tau = tau
        self.median_half_width = median_half_width
        self.sigma_threshold = sigma_threshold

    def warmup(self):
        """Pré-compile tous les noyaux JIT Numba à froid."""
        dummy = np.random.normal(0, 1.0, 1024)
        self.process_stream(dummy)

    def process_stream(self, raw_stream: np.ndarray) -> dict:
        """
        Analyse un flux entrant, extrait la composante d'orchestration spectrale,
        reconstruit le signal organique et calcule la télémétrie avant/après.
        """
        n = raw_stream.shape[0]

        # 1. Passage en espace de Fourier et profilage spectral brut
        spec_raw = rfft(raw_stream)
        freqs = rfftfreq(n, d=1.0 / self.fs)
        psd_raw = (np.abs(spec_raw) ** 2) / float(n)
        s_ent_raw, s_kurt_raw = _spectral_moments(psd_raw[1:])

        # 2. Détection et suppression chirurgicale des fréquences de pompage
        gains, anomalies, pump_ratio = _adaptive_notch_gains(
            psd_raw, self.median_half_width, self.sigma_threshold
        )
        spec_clean = spec_raw * gains
        psd_clean = (np.abs(spec_clean) ** 2) / float(n)
        s_ent_clean, s_kurt_clean = _spectral_moments(psd_clean[1:])

        # 3. Reconstruction temporelle en phase nulle
        clean_stream = irfft(spec_clean, n=n)
        orchestrated_wave = raw_stream - clean_stream
        pumping_freqs = freqs[np.where(anomalies == 1)[0]]

        # 4. Télémétrie Tsallis + Espace des phases sur flux brut vs flux purifié
        sq_raw, dsq_raw, d2_raw, cer_raw = _sliding_manifold_telemetry(
            raw_stream, self.window_size, self.step,
            self.tsallis_q, self.n_bins, self.embed_dim, self.tau
        )
        sq_cln, dsq_cln, d2_cln, cer_cln = _sliding_manifold_telemetry(
            clean_stream, self.window_size, self.step,
            self.tsallis_q, self.n_bins, self.embed_dim, self.tau
        )

        return {
            "signals": {
                "raw": raw_stream,
                "clean": clean_stream,
                "orchestrated": orchestrated_wave,
            },
            "spectral": {
                "pumping_freqs_hz": pumping_freqs,
                "pumping_energy_ratio": pump_ratio,
                "raw_entropy": s_ent_raw,
                "clean_entropy": s_ent_clean,
                "raw_kurtosis": s_kurt_raw,
                "clean_kurtosis": s_kurt_clean,
            },
            "manifold_raw": {
                "tsallis_sq": sq_raw,
                "delta_sq": dsq_raw,
                "attractor_d2": d2_raw,
                "cer": cer_raw,
            },
            "manifold_clean": {
                "tsallis_sq": sq_cln,
                "delta_sq": dsq_cln,
                "attractor_d2": d2_cln,
                "cer": cer_cln,
            }
        }


# ============================================================================
# BANC DE VALIDATION EXPÉRIMENTAL
# ============================================================================

if __name__ == "__main__":
    np.random.seed(2026)
    FS = 100.0
    N_POINTS = 16384
    t = np.arange(N_POINTS) / FS

    # A. Génération d'un signal organique complexe (Bruit rose 1/f + dynamique lente)
    w_fft = rfft(np.random.normal(0, 1.0, N_POINTS))
    f_ax = rfftfreq(N_POINTS, d=1.0 / FS)
    pink_env = np.where(f_ax > 0, 1.0 / (f_ax ** 0.45), 0.0)
    organic_core = irfft(w_fft * pink_env, n=N_POINTS)
    organic_core = (organic_core / np.std(organic_core)) + 0.65 * np.sin(2 * np.pi * 0.22 * t)

    # B. Génération d'un forçage d'orchestration (Porteuses + rafales non-linéaires)
    forcing_wave = (
        1.55 * np.sin(2 * np.pi * 3.75 * t)
        + 1.20 * np.cos(2 * np.pi * 9.60 * t + 0.3)
        + 0.95 * np.sin(2 * np.pi * 18.25 * t)
        + 0.70 * np.sin(2 * np.pi * 31.00 * t)
    )
    input_stream = organic_core + forcing_wave

    # Initialisation et pré-compilation d'AETHER-SIEVE
    engine = AetherSieveEngine(fs=FS, window_size=512, step=128, tsallis_q=1.45)
    engine.warmup()

    # Exécution chronométrée
    t_start = time.perf_counter()
    report = engine.process_stream(input_stream)
    exec_ms = (time.perf_counter() - t_start) * 1000.0

    # Corrélation de fidélité avec le signal organique d'origine
    fidelity_raw = np.corrcoef(organic_core, input_stream)[0, 1]
    fidelity_clean = np.corrcoef(organic_core, report["signals"]["clean"])[0, 1]

    sp = report["spectral"]
    m_raw = report["manifold_raw"]
    m_cln = report["manifold_clean"]

    print("=" * 78)
    print(f" AETHER-SIEVE v1.0 | Rapport d'Isolation & Télémétrie ({N_POINTS} éch. en {exec_ms:.2f} ms)")
    print("=" * 78)
    print(f" [SPECTRAL] Porteuses neutralisées (Hz) : {np.round(sp['pumping_freqs_hz'], 2)}")
    print(f" [SPECTRAL] Énergie artificielle purgée  : {sp['pumping_energy_ratio'] * 100:.2f} %")
    print("-" * 78)
    print(f" {'Métrique de Système':<34} | {'Flux Brut (Subi)':<18} | {'Flux Purifié (Isolé)':<18}")
    print("-" * 78)
    print(f" {'Entropie Spectrale (0->1)':<34} | {sp['raw_entropy']:<18.4f} | {sp['clean_entropy']:<18.4f}")
    print(f" {'Kurtosis Spectral (Concentration)':<34} | {sp['raw_kurtosis']:<18.2f} | {sp['clean_kurtosis']:<18.2f}")
    print(f" {'Entropie de Tsallis Moyenne (Sq)':<34} | {np.mean(m_raw['tsallis_sq']):<18.4f} | {np.mean(m_cln['tsallis_sq']):<18.4f}")
    print(f" {'Instabilité Delta Sq (|dSq|)':<34} | {np.mean(np.abs(m_raw['delta_sq'])):<18.4f} | {np.mean(np.abs(m_cln['delta_sq'])):<18.4f}")
    print(f" {'Dimension Attracteur Takens (D2)':<34} | {np.mean(m_raw['attractor_d2']):<18.4f} | {np.mean(m_cln['attractor_d2']):<18.4f}")
    print(f" {'Ratio Émergence Critique (CER)':<34} | {np.mean(m_raw['cer']):<18.4f} | {np.mean(m_cln['cer']):<18.4f}")
    print("-" * 78)
    print(f" {'Corrélation au Signal Réel Pur':<34} | {fidelity_raw:<18.4f} | {fidelity_clean:<18.4f}")
    print("=" * 78)
