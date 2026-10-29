from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# UMAP est optionnel (meilleure préservation des structures locales)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("umap-learn non installé → fallback PCA")

@dataclass
class OrbitPoint:
    t: int
    features: np.ndarray          # (6,)
    coords_3d: Optional[np.ndarray] = None
    nearest_attractor: Optional[str] = None
    spectral_snapshot: Optional[np.ndarray] = None

@dataclass
class InstanceOrbit:
    orbit_id: str
    prompt_history: List[str]
    points: List[OrbitPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class NavigationLayer:
    """
    Couche de navigation 3D + orbites d'instance.
    Strictement en lecture seule par rapport à l'Atlas.
    """

    def __init__(
        self,
        atlas,                          # instance de NonExtensiveAtlas (lecture seule)
        projection_dim: int = 3,
        use_umap: bool = True,
        random_state: int = 42
    ):
        self.atlas = atlas
        self.projection_dim = projection_dim
        self.use_umap = use_umap and HAS_UMAP
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.projector = None
        self.is_fitted = False

        self.orbits: Dict[str, InstanceOrbit] = {}
        self.feature_names = [
            "S_0.7", "S_1.0", "S_1.5", "production", "d_nearest", "residual_norm"
        ]

        # Pré-extraction des centroïdes de l'Atlas pour le calcul de d_nearest
        self._atlas_centroids = self._extract_atlas_centroids()

    def _extract_atlas_centroids(self) -> Dict[str, np.ndarray]:
        """Récupère les centroïdes mid-layer de l'Atlas (lecture seule)."""
        centroids = {}
        mid_layer = sorted(self.atlas.layers_to_track)[len(self.atlas.layers_to_track)//2]
        for att_id, att in self.atlas.attractors.items():
            if mid_layer in att.centroid_layers:
                centroids[att_id] = att.centroid_layers[mid_layer]
        return centroids

    def _compute_features(
        self,
        spectral: np.ndarray,          # (T, n_q)
        production: np.ndarray,        # (T-1, n_q)
        residuals_mid: np.ndarray,     # (T, d_model)
        q_grid: np.ndarray
    ) -> np.ndarray:
        """
        Transforme une trajectoire brute en série de vecteurs de features minimales.
        Retourne array de shape (T, 6)
        """
        T = spectral.shape[0]
        q_idx = {
            0.7: np.argmin(np.abs(q_grid - 0.7)),
            1.0: np.argmin(np.abs(q_grid - 1.0)),
            1.5: np.argmin(np.abs(q_grid - 1.5))
        }

        features = np.zeros((T, 6))

        for t in range(T):
            features[t, 0] = spectral[t, q_idx[0.7]]
            features[t, 1] = spectral[t, q_idx[1.0]]
            features[t, 2] = spectral[t, q_idx[1.5]]

            # Production (moyenne sur les 3 q, avec padding au premier pas)
            if t == 0 or production.shape[0] == 0:
                features[t, 3] = 0.0
            else:
                idx = min(t-1, production.shape[0]-1)
                features[t, 3] = production[idx, [q_idx[0.7], q_idx[1.0], q_idx[1.5]]].mean()

            # Distance au centroïde le plus proche
            if len(self._atlas_centroids) > 0 and t < residuals_mid.shape[0]:
                dists = [
                    np.linalg.norm(residuals_mid[t] - c)
                    for c in self._atlas_centroids.values()
                ]
                features[t, 4] = min(dists) if dists else 0.0
            else:
                features[t, 4] = 0.0

            # Norme du residual (normalisée plus tard par le scaler)
            if t < residuals_mid.shape[0]:
                features[t, 5] = np.linalg.norm(residuals_mid[t])
            else:
                features[t, 5] = 0.0

        return features

    def fit_projector(self, list_of_feature_series: List[np.ndarray]):
        """
        Ajuste le projecteur (UMAP ou PCA) sur un ensemble de trajectoires.
        À appeler une fois avec un bon échantillon diversifié.
        """
        all_points = np.vstack(list_of_feature_series)
        self.scaler.fit(all_points)
        scaled = self.scaler.transform(all_points)

        if self.use_umap:
            self.projector = umap.UMAP(
                n_components=self.projection_dim,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=self.random_state
            )
        else:
            self.projector = PCA(
                n_components=self.projection_dim,
                random_state=self.random_state
            )

        self.projector.fit(scaled)
        self.is_fitted = True
        print(f"[Navigation] Projecteur ajusté ({'UMAP' if self.use_umap else 'PCA'})")

    def create_orbit(
        self,
        orbit_id: str,
        trajectory: Dict[str, Any],
        prompt: str,
        mid_layer: Optional[int] = None
    ) -> InstanceOrbit:
        """
        Crée une orbite d'instance à partir d'une trajectoire générée par l'Atlas.
        """
        if mid_layer is None:
            mid_layer = sorted(self.atlas.layers_to_track)[len(self.atlas.layers_to_track)//2]

        spectral = trajectory["spectral"]
        production = trajectory.get("production", np.zeros((0, spectral.shape[1])))
        residuals_mid = trajectory["residuals"][mid_layer]

        features = self._compute_features(
            spectral, production, residuals_mid, self.atlas.q_config.q_grid
        )

        orbit = InstanceOrbit(
            orbit_id=orbit_id,
            prompt_history=[prompt],
            metadata={"n_steps": features.shape[0]}
        )

        # Projection si le projecteur est prêt
        coords = None
        if self.is_fitted:
            scaled = self.scaler.transform(features)
            coords = self.projector.transform(scaled)

        for t in range(features.shape[0]):
            nearest = None
            if len(self._atlas_centroids) > 0:
                # simple nearest
                dists = {
                    att: np.linalg.norm(residuals_mid[t] - c)
                    for att, c in self._atlas_centroids.items()
                }
                nearest = min(dists, key=dists.get)

            point = OrbitPoint(
                t=t,
                features=features[t],
                coords_3d=coords[t] if coords is not None else None,
                nearest_attractor=nearest,
                spectral_snapshot=spectral[t]
            )
            orbit.points.append(point)

        self.orbits[orbit_id] = orbit
        return orbit

    def get_orbit_coordinates(self, orbit_id: str) -> np.ndarray:
        """Retourne les coordonnées 3D d'une orbite (T, 3)."""
        orbit = self.orbits[orbit_id]
        if not orbit.points or orbit.points[0].coords_3d is None:
            raise RuntimeError("Orbite non projetée. Appelez fit_projector puis recreate.")
        return np.stack([p.coords_3d for p in orbit.points])

    def overlay_orbits(
        self,
        orbit_ids: List[str],
        include_atlas_basins: bool = True
    ) -> Dict[str, Any]:
        """
        Prépare les données de superposition pour visualisation.
        Ne génère pas encore le plot (responsabilité du visualizer).
        """
        data = {
            "orbits": {},
            "atlas_basins": {}
        }

        for oid in orbit_ids:
            if oid in self.orbits:
                coords = self.get_orbit_coordinates(oid)
                data["orbits"][oid] = {
                    "coords": coords.tolist(),
                    "nearest": [p.nearest_attractor for p in self.orbits[oid].points],
                    "prompts": self.orbits[oid].prompt_history
                }

        if include_atlas_basins and self.is_fitted:
            # Projection approximative des centroïdes (via features synthétiques)
            # Pour une vraie intégration on construirait des features de centroïde
            pass  # laissé volontairement simple pour l'instant

        return data

    def save_orbits(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for oid, orbit in self.orbits.items():
            serializable[oid] = {
                "orbit_id": orbit.orbit_id,
                "prompt_history": orbit.prompt_history,
                "points": [
                    {
                        "t": p.t,
                        "features": p.features.tolist(),
                        "coords_3d": p.coords_3d.tolist() if p.coords_3d is not None else None,
                        "nearest_attractor": p.nearest_attractor
                    } for p in orbit.points
                ],
                "metadata": orbit.metadata
            }
        with open(path / "orbits.json", "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[Navigation] {len(self.orbits)} orbites sauvegardées.")
