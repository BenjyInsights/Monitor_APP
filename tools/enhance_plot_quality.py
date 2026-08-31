#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enhance_plot_quality.py

Module for generating publication-ready plots with professional styling.

Features:
  ✓ 300 DPI output for print quality
  ✓ Print-friendly color palettes (colorblind-safe)
  ✓ Professional typography with sans-serif fonts
  ✓ High contrast graphics suitable for PDF and grayscale printing
  ✓ Consistent UNED institutional branding

Usage:
    from enhance_plot_quality import PublicationPlotter
    
    plotter = PublicationPlotter(dpi=300, style="seaborn-v0_8-darkgrid")
    plotter.save_figure("output.pdf")

Author: Senior DevOps Engineer
Date: 2026-04-16
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from analysis_utils import UNED_COLORS, MODEL_COLORS


class PublicationPlotter:
    """Helper class for generating publication-ready plots."""

    # Publication standards
    DPI_PUBLICATION = 300
    DPI_SCREEN = 72
    
    # Figsize standards (inches)
    FIGSIZE_FULL_PAGE = (16, 10)
    FIGSIZE_HALF_PAGE = (10, 6)
    FIGSIZE_COLUMN = (8, 5)

    def __init__(
        self,
        figsize: Tuple[float, float] = FIGSIZE_HALF_PAGE,
        dpi: int = DPI_PUBLICATION,
        style: str = "seaborn-v0_8-whitegrid",
        font_family: str = "DejaVu Sans",
    ):
        """
        Initialize publication-quality plot styling.

        Parameters
        ----------
        figsize : tuple
            Figure size in inches (width, height).
        dpi : int
            Dots per inch for rasterization. 300 for print, 72 for screen.
        style : str
            Seaborn style name.
        font_family : str
            Default font family (sans-serif for publications).
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.font_family = font_family
        self.fig = None
        self.ax = None
        
        # Apply styling
        self._apply_publication_style()

    def _apply_publication_style(self) -> None:
        """Apply professional styling to matplotlib and seaborn."""
        # Set style
        if self.style:
            try:
                sns.set_style(self.style)
            except:
                sns.set_style("whitegrid")

        # Global matplotlib settings
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.sans-serif'] = [self.font_family, 'Helvetica', 'Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
        
        # Line and patch properties
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['lines.markersize'] = 6

        # Ensure vector output where possible
        plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF
        plt.rcParams['ps.fonttype'] = 42

    def create_figure(self, title: str = "", **kwargs) -> Tuple:
        """
        Create a new figure with publication defaults.

        Parameters
        ----------
        title : str
            Figure title (optional).
        **kwargs
            Additional arguments to plt.subplots().

        Returns
        -------
        tuple
            (fig, ax) from matplotlib.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize, **kwargs)
        
        if title:
            self.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        return self.fig, self.ax

    def add_uned_watermark(self, ax=None, alpha: float = 0.1) -> None:
        """Add subtle UNED institutional watermark (optional, for internal use)."""
        ax = ax or self.ax
        if ax:
            ax.text(
                0.98, 0.02,
                "© 2026 moniaenergy",
                transform=ax.transAxes,
                fontsize=7,
                alpha=alpha,
                ha='right',
                va='bottom',
                color='gray',
            )

    def save_figure(
        self,
        filepath: Path,
        formats: List[str] = ["pdf"],
        transparent: bool = False,
        bbox_inches: str = "tight",
        pad_inches: float = 0.2,
    ) -> None:
        """
        Save figure in publication-ready format.

        Parameters
        ----------
        filepath : Path
            Output path (without extension).
        formats : list
            File formats to save ["pdf", "png", "svg"].
        transparent : bool
            Whether background should be transparent.
        bbox_inches : str
            Bounding box setting ("tight" removes white space).
        pad_inches : float
            Padding around the figure in inches.
        """
        if self.fig is None:
            raise RuntimeError("No figure to save. Call create_figure() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            output = filepath.with_suffix(f".{fmt}")
            
            # Select DPI based on format
            dpi = self.DPI_PUBLICATION if fmt in ["pdf", "svg"] else self.DPI_PUBLICATION
            
            self.fig.savefig(
                output,
                format=fmt,
                dpi=dpi,
                transparent=transparent,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                facecolor='white' if not transparent else 'none',
            )
            print(f"✓ Saved: {output}")

    def configure_colorbar(self, im, label: str = "", ax=None) -> None:
        """Configure a colorbar with publication styling."""
        ax = ax or self.ax
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=9)

    @staticmethod
    def get_uned_palette(n_colors: int = 10) -> List[str]:
        """
        Get a publication-grade color palette from UNED colors.

        Parameters
        ----------
        n_colors : int
            Number of colors needed.

        Returns
        -------
        list
            List of hex color codes.
        """
        base_colors = list(UNED_COLORS.values())
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        
        # Interpolate additional colors if needed
        palette = base_colors.copy()
        while len(palette) < n_colors:
            palette.extend(base_colors)
        return palette[:n_colors]

    @staticmethod
    def get_colorblind_safe_palette() -> Dict[str, str]:
        """
        Return a colorblind-safe palette suitable for color-deficient viewers.

        Based on:
        - Okabe, M., & Ito, K. (2008). "Color Universal Design (CUD):
          How to make figures and presentations that are friendly to colorblind people"

        Returns
        -------
        dict
            Mapping of descriptive names to hex colors.
        """
        return {
            'blue': '#0173B2',
            'orange': '#DE8F05',
            'green': '#029E73',
            'red': '#CC78BC',
            'purple': '#CA9161',
            'gray': '#949494',
            'black': '#ECE2F0',
        }


# ============================================================================
# Convenience Functions for Common Plot Types
# ============================================================================

def create_pareto_plot(
    df,
    x_col: str,
    y_col: str,
    label_col: str = None,
    color_col: str = None,
    output_path: Optional[Path] = None,
    title: str = "Pareto Frontier: Energy vs. Accuracy",
) -> None:
    """
    Create a high-quality Pareto frontier plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns for x, y, and optional label/color.
    x_col : str
        Column name for x-axis (typically energy).
    y_col : str
        Column name for y-axis (typically accuracy).
    label_col : str, optional
        Column for point labels (model names).
    color_col : str, optional
        Column for point colors (architectures).
    output_path : Path, optional
        Save path for the figure.
    title : str
        Plot title.
    """
    plotter = PublicationPlotter(figsize=PublicationPlotter.FIGSIZE_HALF_PAGE)
    fig, ax = plotter.create_figure(title=title)

    # Prepare colors
    if color_col and color_col in df.columns:
        unique_colors = df[color_col].unique()
        colors = {col: MODEL_COLORS.get(col, UNED_COLORS['BLUE']) for col in unique_colors}
        for color_val in unique_colors:
            mask = df[color_col] == color_val
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=color_val,
                color=colors[color_val],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5,
            )
    else:
        ax.scatter(
            df[x_col],
            df[y_col],
            color=UNED_COLORS['UNED'],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
        )

    # Compute and plot Pareto frontier
    pareto_mask = df[[x_col, y_col]].apply(
        lambda row: not any(
            (df[x_col] < row[x_col]) & (df[y_col] > row[y_col])
        ), axis=1
    )
    
    if pareto_mask.any():
        pareto_df = df[pareto_mask].sort_values(x_col)
        ax.plot(
            pareto_df[x_col],
            pareto_df[y_col],
            color=UNED_COLORS['RASPBERRY'],
            linewidth=2.0,
            linestyle='--',
            label='Pareto Frontier',
            zorder=10,
        )

    # Labels and formatting
    ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.95)

    if output_path:
        plotter.save_figure(output_path, formats=['pdf', 'png'])
    else:
        plt.show()

    plt.close(fig)


def create_bar_comparison(
    data: Dict[str, float],
    title: str = "Energy Comparison",
    ylabel: str = "Energy (J)",
    output_path: Optional[Path] = None,
) -> None:
    """
    Create a professional bar chart for metric comparison.

    Parameters
    ----------
    data : dict
        Mapping of labels to values.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    output_path : Path, optional
        Save path for the figure.
    """
    plotter = PublicationPlotter(figsize=PublicationPlotter.FIGSIZE_COLUMN)
    fig, ax = plotter.create_figure(title=title)

    labels = list(data.keys())
    values = list(data.values())
    colors_list = plotter.get_uned_palette(len(labels))

    bars = ax.bar(labels, values, color=colors_list, edgecolor='black', linewidth=1.0)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    plt.xticks(rotation=45, ha='right')

    if output_path:
        plotter.save_figure(output_path, formats=['pdf', 'png'])
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    print("enhance_plot_quality module loaded successfully.")
    print("Use PublicationPlotter class for creating high-quality figures.")
