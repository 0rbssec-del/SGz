from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorsys
from pathlib import Path
import json

class AtlasNavigator3D:
    """
    Visualiseur 3D technique et professionnel des orbites d'instance
    superposées à l'Atlas non-extensif.
    """

    def __init__(
        self,
        dark_mode: bool = True,
        width: int = 1280,
        height: int = 900,
        title: str = "Non-Extensive Manifold Navigator | Qwen3-8B"
    ):
        self.dark_mode = dark_mode
        self.width = width
        self.height = height
        self.title = title

        # Palette technique
        self.bg_color = "#0d1117" if dark_mode else "#f8f9fa"
        self.grid_color = "#21262d" if dark_mode else "#dee2e6"
        self.font_color = "#e6edf3" if dark_mode else "#212529"
        self.axis_color = "#8b949e"

        # Couleurs d'orbites (générées dynamiquement)
        self.orbit_colors = self._generate_orbit_palette(12)

        # Couleur des attracteurs Atlas
        self.attractor_color = "#ff7b72"      # rouge technique
        self.attractor_line = "#ffa198"

    def _generate_orbit_palette(self, n: int) -> List[str]:
        """Génère une palette de couleurs distinctes pour les orbites."""
        colors = []
        for i in range(n):
            hue = i / n
            # Saturation et valeur élevées pour fond sombre
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
            colors.append(
                f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"
            )
        return colors

    def _create_base_layout(self) -> go.Layout:
        """Layout technique sombre / clair."""
        return go.Layout(
            title=dict(
                text=self.title,
                font=dict(size=18, color=self.font_color, family="JetBrains Mono, monospace"),
                x=0.02,
                y=0.98
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.font_color, family="Inter, sans-serif", size=12),
            width=self.width,
            height=self.height,
            margin=dict(l=40, r=40, t=60, b=40),
            scene=dict(
                xaxis=dict(
                    title="Axis-1 (Spectral / q-structure)",
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    zerolinecolor=self.axis_color,
                    color=self.axis_color,
                    titlefont=dict(size=11)
                ),
                yaxis=dict(
                    title="Axis-2 (Production / Irreversibility)",
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    zerolinecolor=self.axis_color,
                    color=self.axis_color,
                    titlefont=dict(size=11)
                ),
                zaxis=dict(
                    title="Axis-3 (Distance to Attractors)",
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    zerolinecolor=self.axis_color,
                    color=self.axis_color,
                    titlefont=dict(size=11)
                ),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.25),
                    center=dict(x=0, y=0, z=-0.05)
                )
            ),
            legend=dict(
                bgcolor="rgba(13,17,23,0.85)" if self.dark_mode else "rgba(255,255,255,0.9)",
                bordercolor=self.grid_color,
                borderwidth=1,
                font=dict(size=11),
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top"
            ),
            hovermode="closest"
        )

    def render(
        self,
        overlay_data: Dict[str, Any],
        show_attractor_labels: bool = True,
        show_time_gradient: bool = True,
        marker_size_orbit: float = 3.5,
        line_width: float = 2.0,
        save_html: Optional[str] = None,
        auto_open: bool = False
    ) -> go.Figure:
        """
        Construit et retourne la figure 3D interactive.

        Parameters
        ----------
        overlay_data : dict
            Sortie de NavigationLayer.overlay_orbits()
        """
        fig = go.Figure()

        # ------------------------------------------------------------------
        # 1. Orbites d'instance
        # ------------------------------------------------------------------
        orbit_ids = list(overlay_data.get("orbits", {}).keys())

        for idx, oid in enumerate(orbit_ids):
            orbit = overlay_data["orbits"][oid]
            coords = np.array(orbit["coords"])          # (T, 3)
            nearest = orbit.get("nearest", [None] * len(coords))

            color = self.orbit_colors[idx % len(self.orbit_colors)]

            # Ligne de trajectoire
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="lines",
                line=dict(
                    color=color,
                    width=line_width
                ),
                name=f"Orbit {oid}",
                legendgroup=oid,
                hoverinfo="skip",
                opacity=0.85
            ))

            # Points avec gradient temporel optionnel
            if show_time_gradient:
                # Couleur du point = mélange entre couleur d'orbite et blanc selon le temps
                t_norm = np.linspace(0, 1, len(coords))
                point_colors = [
                    self._blend_color(color, "#ffffff", 0.15 + 0.55 * t)
                    for t in t_norm
                ]
            else:
                point_colors = color

            # Texte hover riche
            hover_texts = []
            for t, (x, y, z), near in zip(range(len(coords)), coords, nearest):
                hover_texts.append(
                    f"<b>Orbit</b>: {oid}<br>"
                    f"<b>Step</b>: {t}<br>"
                    f"<b>Nearest Attractor</b>: {near or '—'}<br>"
                    f"X: {x:.3f} | Y: {y:.3f} | Z: {z:.3f}"
                )

            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size_orbit,
                    color=point_colors if show_time_gradient else color,
                    opacity=0.95,
                    line=dict(width=0.5, color="rgba(255,255,255,0.25)")
                ),
                name=f"Points {oid}",
                legendgroup=oid,
                showlegend=False,
                text=hover_texts,
                hoverinfo="text"
            ))

            # Marqueur de départ (t=0)
            fig.add_trace(go.Scatter3d(
                x=[coords[0, 0]],
                y=[coords[0, 1]],
                z=[coords[0, 2]],
                mode="markers",
                marker=dict(
                    size=9,
                    color=color,
                    symbol="diamond",
                    line=dict(width=1.5, color="#ffffff")
                ),
                name=f"Start {oid}",
                legendgroup=oid,
                showlegend=False,
                hovertext=f"<b>START</b> — {oid}",
                hoverinfo="text"
            ))

        # ------------------------------------------------------------------
        # 2. Attracteurs de l'Atlas (si fournis)
        # ------------------------------------------------------------------
        basins = overlay_data.get("atlas_basins", {})
        if basins:
            for att_id, basin in basins.items():
                # On attend un centroïde 3D approximatif
                c = np.array(basin.get("coords", [0, 0, 0]))
                fig.add_trace(go.Scatter3d(
                    x=[c[0]],
                    y=[c[1]],
                    z=[c[2]],
                    mode="markers+text" if show_attractor_labels else "markers",
                    marker=dict(
                        size=14,
                        color=self.attractor_color,
                        symbol="x",
                        line=dict(width=2, color=self.attractor_line)
                    ),
                    text=[att_id] if show_attractor_labels else None,
                    textposition="top center",
                    textfont=dict(size=10, color=self.attractor_color),
                    name=f"Attractor {att_id}",
                    hovertext=f"<b>ATTRACTOR</b><br>{att_id}<br>Stability: {basin.get('stability', 'n/a')}",
                    hoverinfo="text"
                ))

        # ------------------------------------------------------------------
        # 3. Layout final
        # ------------------------------------------------------------------
        fig.update_layout(self._create_base_layout())

        # Annotation technique en bas
        fig.add_annotation(
            text="Non-Extensive Geometry • Tsallis Coordinates • Read-only overlay on Atlas Core",
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=10, color="#6e7681"),
            align="left"
        )

        if save_html:
            path = Path(save_html)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(
                str(path),
                include_plotlyjs="cdn",
                full_html=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["toImage"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "atlas_navigator_3d",
                        "height": self.height,
                        "width": self.width,
                        "scale": 2
                    }
                }
            )
            print(f"[Navigator3D] Figure sauvegardée → {path}")

        if auto_open:
            fig.show()

        return fig

    def _blend_color(self, color1: str, color2: str, t: float) -> str:
        """Mélange linéaire simple entre deux couleurs rgb/hex."""
        def parse(c):
            if c.startswith("rgb"):
                vals = c[4:-1].split(",")
                return [int(v) for v in vals]
            # hex
            c = c.lstrip("#")
            return [int(c[i:i+2], 16) for i in (0, 2, 4)]

        r1, g1, b1 = parse(color1)
        r2, g2, b2 = parse(color2)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"rgb({r},{g},{b})"

    def render_from_file(
        self,
        overlay_json_path: str,
        **kwargs
    ) -> go.Figure:
        """Charge un fichier overlay JSON et rend la figure."""
        with open(overlay_json_path, "r") as f:
            data = json.load(f)
        return self.render(data, **kwargs)
