"""
Visualization Functions
=======================

Publication-quality figure generation for TUS gene clustering analysis.

FIGURE INVENTORY:
=================

Main Figures:
    Figure 1 Main: Combined clustering overview with gene profiles
              - Heatmap with hierarchical gene ordering
              - 3D brain renders, glass brains, axial slices
              - Radar plots showing gene expression patterns
              - Output: output/figures/figure1_main.pdf

    Figure 1 Legacy: Clustering Overview (create_figure1_clustering)
              - Heatmaps at K_OPTIMAL and K_GRANULAR
              - Brain slice visualizations
              - Output: output/figures/figure1_clustering.pdf

Supplemental Figures:
    Supp 1: K_OPTIMAL Clustering Overview
            - Output: output/figures/supplemental1_optimal.pdf

    Supp 2: Silhouette Curve (create_supplemental_robust_silhouette)
            - Robust silhouette analysis with error bars
            - Output: output/figures/supplemental_silhouette.pdf

    Supp 3: Monte Carlo ARI (create_figure2_monte_carlo)
            - ARI distributions comparing TUS vs random genes
            - Output: output/figures/supplemental_monte_carlo.pdf

Color Scheme:
    - Heatmap: Neo Tokyo (hot pink → dark → cyan)
    - Cluster colors defined in config.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram
from . import config


# =============================================================================
# HEATMAP COLORMAP OPTIONS
# =============================================================================

# ACTIVE: Neo Tokyo (flipped: cyan → dark → pink)
# Cyan (low) → dark at 0 → hot pink (high)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'neo_tokyo',
    [(0.0, '#01fff4'), (0.3, '#05d9e8'), (0.5, '#1a1a2e'), (1.0, '#ff2a6d')]
)

# ARCHIVED OPTIONS (commented out)
# Neo Tokyo Wide: ['#ff2a6d', '#1a1a2e', '#0a0a14', '#0a0a14', '#05d9e8', '#01fff4']
# Neon Nights: ['#4a0080', '#2d004d', '#1a1a2e', '#1a1a2e', '#ff0080', '#ff4da6', '#ff99cc']
# Sunset Grid: ['#2d1b4e', '#6b2d5b', '#d63384', '#ff6b35', '#f7b32b', '#fcf6b1']


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.TABLES_DIR, exist_ok=True)


def generate_shade(hex_color, factor):
    """
    Generate a lighter or darker shade of a hex color.

    Args:
        hex_color: Hex color string (e.g., '#2e5a8c')
        factor: <1 for darker, >1 for lighter (e.g., 0.7 = 30% darker, 1.3 = 30% lighter)

    Returns:
        Hex color string of the shade
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    if factor < 1:
        # Darken: multiply by factor
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
    else:
        # Lighten: blend toward white
        blend = factor - 1
        r = int(r + (255 - r) * blend)
        g = int(g + (255 - g) * blend)
        b = int(b + (255 - b) * blend)

    # Clamp values
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

    return f'#{r:02x}{g:02x}{b:02x}'


def generate_granular_colors_from_parent_map(k_gran_to_k_opt_map, opt_colors):
    """
    Generate K_GRANULAR colors as shades of their K_OPTIMAL parent colors.

    Args:
        k_gran_to_k_opt_map: Dict mapping granular cluster index -> optimal parent index
        opt_colors: List of K_OPTIMAL hex colors

    Returns:
        List of hex colors (shades based on parent)
    """
    # Count how many subclusters each parent has
    parent_counts = {}
    for gran_idx, opt_parent in k_gran_to_k_opt_map.items():
        parent_counts[opt_parent] = parent_counts.get(opt_parent, 0) + 1

    # Track which shade index we're on for each parent
    parent_shade_idx = {p: 0 for p in range(len(opt_colors))}

    # Shade factors: for n subclusters, spread from dark to light
    def get_shade_factors(n):
        if n == 1:
            return [1.0]  # Base color
        elif n == 2:
            return [0.7, 1.0]  # Dark, base
        elif n == 3:
            return [0.6, 1.0, 1.4]  # Dark, base, light
        else:
            # Spread evenly from 0.5 to 1.5
            return [0.5 + i * (1.0 / (n - 1)) for i in range(n)]

    # Pre-compute shade factors for each parent
    shade_factors = {p: get_shade_factors(parent_counts.get(p, 1)) for p in range(len(opt_colors))}

    # Generate colors for each granular cluster
    gran_colors = []
    for gran_idx in range(len(k_gran_to_k_opt_map)):
        opt_parent = k_gran_to_k_opt_map[gran_idx]
        shade_idx = parent_shade_idx[opt_parent]
        factor = shade_factors[opt_parent][shade_idx]

        gran_colors.append(generate_shade(opt_colors[opt_parent], factor))
        parent_shade_idx[opt_parent] += 1

    return gran_colors


# =============================================================================
# FIGURE 1: CLUSTERING OVERVIEW
# =============================================================================

def create_heatmap_panel(fig, gs_heatmap, cluster_means, k, cluster_colors,
                         gene_linkage, genes, panel_label, title,
                         row_linkage=None, parent_colors=None, gran_to_opt_map=None,
                         cmap=None):
    """
    Create a heatmap panel with dendrogram(s).

    Layout:
        - Column dendrogram on TOP (hierarchical clustering of genes)
        - Row dendrogram on LEFT (optional, for hierarchical clustering of clusters)
        - Heatmap: Clusters as ROWS, Genes as COLUMNS
        - Colorbar on RIGHT
        - Parent color boxes on dendrogram (if gran_to_opt_map provided)

    Args:
        fig: Figure object
        gs_heatmap: GridSpec for this panel
        cluster_means: Mean expression per cluster (k x n_genes)
        k: Number of clusters
        cluster_colors: Colors for each cluster
        gene_linkage: Hierarchical clustering linkage for genes
        genes: List of gene names
        panel_label: Letter for panel (a, b, c, d)
        title: Panel title
        row_linkage: Optional linkage for cluster dendrogram
        parent_colors: Optional dict mapping cluster -> parent cluster color
        gran_to_opt_map: Optional dict mapping K_GRANULAR cluster -> K_OPTIMAL parent index
        cmap: Colormap for heatmap. Default: HEATMAP_CMAP_K4

    Returns:
        gene_order: Gene names in dendrogram order
        row_order: Cluster indices in dendrogram order
    """
    if cmap is None:
        cmap = HEATMAP_CMAP
    # Use consistent 3-column grid layout for both cases
    # This ensures heatmap widths are identical between panels
    gs_inner = gs_heatmap.subgridspec(
        2, 3,
        height_ratios=[0.15, 1],
        width_ratios=[0.06, 1, 0.02],
        hspace=0.02, wspace=0.01
    )

    if row_linkage is not None:
        ax_row_dend = fig.add_subplot(gs_inner[1, 0])
    else:
        # Create empty placeholder to maintain spacing
        ax_row_dend = fig.add_subplot(gs_inner[1, 0])
        ax_row_dend.axis('off')
        ax_row_dend = None  # Set to None so we skip dendrogram drawing

    ax_col_dend = fig.add_subplot(gs_inner[0, 1])
    ax_heatmap = fig.add_subplot(gs_inner[1, 1])
    ax_colorbar = fig.add_subplot(gs_inner[1, 2])

    # Column dendrogram (genes)
    dend_col = dendrogram(
        gene_linkage,
        labels=genes,
        orientation='top',
        leaf_font_size=0,
        color_threshold=0,
        above_threshold_color=config.LINE_COLOR,
        ax=ax_col_dend
    )
    ax_col_dend.axis('off')
    ax_col_dend.text(-0.05, 1.3, panel_label, transform=ax_col_dend.transAxes,
                     fontsize=16, fontweight='bold')
    ax_col_dend.set_title(title, fontsize=11, pad=10)

    gene_order = [genes[i] for i in dend_col['leaves']]
    gene_indices = [genes.index(g) for g in gene_order]

    # Row dendrogram (clusters) - if provided
    row_order = list(range(k))
    if ax_row_dend is not None and row_linkage is not None:
        dend_row = dendrogram(
            row_linkage,
            orientation='left',
            leaf_font_size=0,
            color_threshold=0,
            above_threshold_color=config.LINE_COLOR,
            ax=ax_row_dend
        )
        ax_row_dend.axis('off')
        ax_row_dend.invert_yaxis()
        row_order = dend_row['leaves']

        # Add colored boxes showing K_OPTIMAL parent groupings
        if gran_to_opt_map is not None:
            # Get the dendrogram limits
            xlim = ax_row_dend.get_xlim()
            ylim = ax_row_dend.get_ylim()

            # Each cluster occupies a band of height 10 in dendrogram coordinates
            # (default spacing in scipy dendrogram)
            band_height = 10

            # Find contiguous groups with same K_OPTIMAL parent in dendrogram order
            parent_sequence = [gran_to_opt_map[c] for c in row_order]

            # Identify group boundaries
            groups = []
            start_idx = 0
            current_parent = parent_sequence[0]
            for i in range(1, len(parent_sequence)):
                if parent_sequence[i] != current_parent:
                    groups.append((start_idx, i - 1, current_parent))
                    start_idx = i
                    current_parent = parent_sequence[i]
            groups.append((start_idx, len(parent_sequence) - 1, current_parent))

            # Draw colored rectangles for each group
            for start_idx, end_idx, parent in groups:
                # Calculate y position (dendrogram uses 5, 15, 25, ... for leaves)
                y_bottom = (start_idx * band_height) + 5 - (band_height / 2)
                y_top = (end_idx * band_height) + 5 + (band_height / 2)
                height = y_top - y_bottom

                # Rectangle spans full width of dendrogram
                rect = Rectangle(
                    (xlim[1], y_bottom),  # Start from right edge (dendrogram grows left)
                    xlim[0] - xlim[1],  # Full width
                    height,
                    facecolor=config.CLUSTER_COLORS_OPT[parent],
                    alpha=0.25,
                    edgecolor=config.CLUSTER_COLORS_OPT[parent],
                    linewidth=2,
                    zorder=0  # Behind the dendrogram lines
                )
                ax_row_dend.add_patch(rect)

    # Heatmap - use auto aspect so dendrogram aligns properly
    heatmap_data = cluster_means[row_order, :][:, gene_indices]

    im = ax_heatmap.imshow(
        heatmap_data,
        aspect='auto',
        cmap=cmap,
        vmin=-2.5, vmax=2.5
    )

    # Y-axis labels (clusters)
    ax_heatmap.set_yticks(range(k))
    ax_heatmap.set_yticklabels([f'C{row_order[c]+1}' for c in range(k)], fontsize=9)

    # X-axis labels (genes) - 45 degree rotation with top aligned to column base
    ax_heatmap.set_xticks(range(len(gene_order)))
    ax_heatmap.set_xticklabels(gene_order, rotation=45, fontsize=8,
                                ha='right', rotation_mode='anchor')
    ax_heatmap.tick_params(axis='x', length=0)

    # Colorbar
    plt.colorbar(im, cax=ax_colorbar, label='Z-score')

    return gene_order, row_order


def create_brain_slices(fig, gs_brain, labels, k, cluster_colors, atlas_data, panel_label):
    """
    Create 10 axial brain slices showing cluster spatial distribution.

    Args:
        fig: Figure object
        gs_brain: GridSpec for brain panel
        labels: Cluster assignments for each brain region
        k: Number of clusters
        cluster_colors: Colors for each cluster
        atlas_data: 3D atlas volume
        panel_label: Letter for panel
    """
    # Create cluster map
    cluster_map = np.zeros_like(atlas_data, dtype=float)
    for roi_id in range(1, 333):
        if roi_id <= len(labels):
            cluster_map[atlas_data == roi_id] = labels[roi_id - 1] + 1

    # Set background (0) to NaN for transparency
    cluster_map[cluster_map == 0] = np.nan

    # Custom colormap - cluster colors only (background is transparent via NaN)
    cluster_cmap = ListedColormap(cluster_colors)
    cluster_cmap.set_bad(color='none')  # Transparent background

    # Find z-range
    z_coords = np.where(atlas_data > 0)[2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_slices = np.linspace(z_min + 5, z_max - 5, 10).astype(int)

    # Find global x,y crop bounds across all slices (to keep consistent framing)
    xy_coords = np.where(atlas_data > 0)
    x_min, x_max = xy_coords[0].min(), xy_coords[0].max()
    y_min, y_max = xy_coords[1].min(), xy_coords[1].max()
    # Add small padding
    pad = 3
    x_min, x_max = max(0, x_min - pad), min(atlas_data.shape[0], x_max + pad)
    y_min, y_max = max(0, y_min - pad), min(atlas_data.shape[1], y_max + pad)

    # Create 2x5 grid for 10 slices - 10% smaller with tighter spacing
    gs_slices = gs_brain.subgridspec(2, 5, wspace=-0.35, hspace=0.02)

    for idx, z in enumerate(z_slices):
        row, col = idx // 5, idx % 5
        ax = fig.add_subplot(gs_slices[row, col])
        ax.set_facecolor('none')  # Transparent axis background

        # Crop the slice to brain bounds
        axial_slice = cluster_map[x_min:x_max, y_min:y_max, z].T
        ax.imshow(
            np.rot90(axial_slice, k=2),
            cmap=cluster_cmap,
            vmin=1, vmax=k,  # Start from 1 (clusters are 1-indexed now)
            interpolation='nearest',
            aspect='equal'
        )
        ax.axis('off')
        ax.set_title(f'z={z}', fontsize=7, pad=1)

        if idx == 0:
            ax.text(-0.1, 1.1, panel_label, transform=ax.transAxes,
                    fontsize=16, fontweight='bold')


def create_figure1_clustering(
    cluster_results,
    gene_linkage,
    cluster_linkage_9,
    gran_parent_colors,
    atlas_data,
    genes,
    output_path=None,
    monte_carlo_results=None,
    silhouette_data=None
):
    """
    Create Figure 1: Dual K_OPTIMAL/K_GRANULAR clustering comparison.

    If monte_carlo_results is provided, adds ARI histogram panel at bottom.
    If silhouette_data is provided, adds silhouette curve panel at bottom.

    Args:
        cluster_results: Dict from clustering.get_cluster_results()
        gene_linkage: Hierarchical linkage for genes
        cluster_linkage_9: Hierarchical linkage for K_GRANULAR clusters
        gran_parent_colors: Dict mapping K_GRANULAR cluster -> K_OPTIMAL parent color
        atlas_data: 3D atlas volume
        genes: List of gene names
        output_path: Base path for output (without extension)
        monte_carlo_results: Optional DataFrame with Monte Carlo ARI results
        silhouette_data: Optional tuple of (k_values, silhouettes) for curve

    Returns:
        fig: Matplotlib figure object
    """
    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'figure1_clustering')

    labels_opt = cluster_results[config.K_OPTIMAL]['labels']
    labels_granular = cluster_results[config.K_GRANULAR]['labels']
    sil_opt = cluster_results[config.K_OPTIMAL]['silhouette']
    sil_granular = cluster_results[config.K_GRANULAR]['silhouette']
    cluster_means_opt = cluster_results[config.K_OPTIMAL]['cluster_means']
    cluster_means_granular = cluster_results[config.K_GRANULAR]['cluster_means']

    # Compute K_GRANULAR to K_OPTIMAL parent mapping for visualization
    k_granular_to_k_opt_map = {}
    for c in range(config.K_GRANULAR):
        regions_in_c = np.where(labels_granular == c)[0]
        k_opt_counts = np.bincount(labels_opt[regions_in_c], minlength=config.K_OPTIMAL)
        k_granular_to_k_opt_map[c] = int(np.argmax(k_opt_counts))

    # Generate K_GRANULAR colors as shades of their K_OPTIMAL parent colors
    gran_colors = generate_granular_colors_from_parent_map(k_granular_to_k_opt_map, config.CLUSTER_COLORS_OPT)

    # Determine figure height based on whether we have Monte Carlo results
    if monte_carlo_results is not None:
        fig_height = 9.5  # Taller to accommodate histograms
        heatmap_bottom = 0.28
        brain_bottom = 0.25
    else:
        fig_height = 7
        heatmap_bottom = 0.12
        brain_bottom = 0.08

    # Create figure with separate grids for heatmaps and brains
    # This ensures brain slices have equal size in both panels
    fig = plt.figure(figsize=(15, fig_height))

    # Left column: heatmaps (different heights for K_OPTIMAL vs K_GRANULAR)
    gs_heatmaps = GridSpec(
        2, 1,
        height_ratios=[1, 2.25],  # Ratio of 9/4 clusters
        left=0.06, right=0.52, top=0.92, bottom=heatmap_bottom,
        hspace=0.4
    )

    # Right column: brain slices (EQUAL heights for consistent scale)
    # Expanded to fill more space with tighter vertical spacing
    gs_brains = GridSpec(
        2, 1,
        height_ratios=[1, 1],  # Equal heights!
        left=0.53, right=0.99, top=0.90, bottom=brain_bottom,
        hspace=0.05
    )

    # Panel A: K_OPTIMAL Heatmap
    create_heatmap_panel(
        fig, gs_heatmaps[0],
        cluster_means_opt, config.K_OPTIMAL, config.CLUSTER_COLORS_OPT,
        gene_linkage, genes,
        'a', f'K={config.K_OPTIMAL} (Silhouette={sil_opt:.3f}, optimal)'
    )

    # Panel B: K_OPTIMAL Brain Slices
    create_brain_slices(
        fig, gs_brains[0],
        labels_opt, config.K_OPTIMAL, config.CLUSTER_COLORS_OPT,
        atlas_data, 'b'
    )

    # Panel C: K_GRANULAR Heatmap (with row dendrogram)
    # Uses dynamically generated shade colors based on parent mapping
    create_heatmap_panel(
        fig, gs_heatmaps[1],
        cluster_means_granular, config.K_GRANULAR, gran_colors,
        gene_linkage, genes,
        'c', f'K={config.K_GRANULAR} (local maximum)',
        row_linkage=cluster_linkage_9,
        parent_colors=gran_parent_colors,
        gran_to_opt_map=k_granular_to_k_opt_map
    )

    # Panel D: K_GRANULAR Brain Slices
    # Uses shade colors that match their K_OPTIMAL parent
    create_brain_slices(
        fig, gs_brains[1],
        labels_granular, config.K_GRANULAR, gran_colors,
        atlas_data, 'd'
    )

    # Add color legends on the far right (outside main panels)
    # K_OPTIMAL color legend (aligned with panel b)
    legend_x = 0.995  # Far right edge
    opt_top = 0.88
    opt_box_height = 0.035
    opt_spacing = 0.045

    for i in range(config.K_OPTIMAL):
        y_pos = opt_top - i * opt_spacing
        # Colored square
        rect = Rectangle((legend_x, y_pos), 0.012, opt_box_height,
                         transform=fig.transFigure, facecolor=config.CLUSTER_COLORS_OPT[i],
                         edgecolor='black', linewidth=0.5, clip_on=False)
        fig.patches.append(rect)
        # Label
        fig.text(legend_x + 0.018, y_pos + opt_box_height/2, f'C{i+1}',
                fontsize=8, va='center', ha='left', fontweight='bold')

    # K_GRANULAR color legend (aligned with panel d)
    gran_top = 0.42
    gran_box_height = 0.022
    gran_spacing = 0.028

    for i in range(config.K_GRANULAR):
        y_pos = gran_top - i * gran_spacing
        # Colored square
        rect = Rectangle((legend_x, y_pos), 0.012, gran_box_height,
                         transform=fig.transFigure, facecolor=gran_colors[i],
                         edgecolor='black', linewidth=0.5, clip_on=False)
        fig.patches.append(rect)
        # Label
        fig.text(legend_x + 0.018, y_pos + gran_box_height/2, f'C{i+1}',
                fontsize=7, va='center', ha='left', fontweight='bold')

    # Bottom panels: Silhouette curve and/or Monte Carlo histogram
    has_bottom_panels = silhouette_data is not None or monte_carlo_results is not None

    if has_bottom_panels:
        # Determine number of bottom panels
        n_bottom = (1 if silhouette_data is not None else 0) + (1 if monte_carlo_results is not None else 0)

        gs_bottom = GridSpec(
            1, n_bottom,
            left=0.08, right=0.92, top=0.20, bottom=0.06,
            wspace=0.25
        )

        panel_idx = 0
        panel_label = 'e'

        # Panel E: Silhouette curve (if provided)
        if silhouette_data is not None:
            k_values, silhouettes = silhouette_data
            ax_sil = fig.add_subplot(gs_bottom[panel_idx])

            # Plot the curve
            ax_sil.plot(k_values, silhouettes, 'o-', color='#264653', linewidth=2, markersize=6)

            # Highlight K_OPTIMAL (global max for K>=3)
            k_opt_idx = list(k_values).index(config.K_OPTIMAL)
            ax_sil.scatter([config.K_OPTIMAL], [silhouettes[k_opt_idx]], s=150, c='#2A9D8F',
                          zorder=5, marker='*', edgecolors='white', linewidths=1)
            ax_sil.annotate(f'K={config.K_OPTIMAL}\n(global)',
                           (config.K_OPTIMAL, silhouettes[k_opt_idx]),
                           textcoords='offset points', xytext=(15, 5),
                           fontsize=9, color='#2A9D8F', fontweight='bold')

            # Highlight K_GRANULAR (local max)
            k_gran_idx = list(k_values).index(config.K_GRANULAR)
            ax_sil.scatter([config.K_GRANULAR], [silhouettes[k_gran_idx]], s=150, c='#E76F51',
                          zorder=5, marker='*', edgecolors='white', linewidths=1)
            ax_sil.annotate(f'K={config.K_GRANULAR}\n(local)',
                           (config.K_GRANULAR, silhouettes[k_gran_idx]),
                           textcoords='offset points', xytext=(15, -15),
                           fontsize=9, color='#E76F51', fontweight='bold')

            ax_sil.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax_sil.set_ylabel('Silhouette Score', fontsize=11)
            ax_sil.set_title('Cluster Quality: Global and Local Maxima', fontsize=12)
            ax_sil.set_xticks(k_values)
            ax_sil.grid(True, alpha=0.3)
            ax_sil.text(-0.08, 1.05, panel_label, transform=ax_sil.transAxes,
                       fontsize=16, fontweight='bold')

            panel_idx += 1
            panel_label = 'f'

        # Panel F: Monte Carlo ARI histogram (if provided)
        if monte_carlo_results is not None:
            ax_hist = fig.add_subplot(gs_bottom[panel_idx])

            # Find K_OPTIMAL and K_GRANULAR ARI columns dynamically
            ari_cols = [c for c in monte_carlo_results.columns if c.startswith('ari_k')]
            ari_cols_sorted = sorted(ari_cols, key=lambda x: int(x.replace('ari_k', '')))

            ari_opt_col = ari_cols_sorted[0]  # Smaller K (optimal)
            ari_gran_col = ari_cols_sorted[1]  # Larger K (granular)
            k_opt_label = int(ari_opt_col.replace('ari_k', ''))
            k_gran_label = int(ari_gran_col.replace('ari_k', ''))

            ari_opt = monte_carlo_results[ari_opt_col].values
            ari_gran = monte_carlo_results[ari_gran_col].values

            mean_ari_opt = ari_opt.mean()
            mean_ari_gran = ari_gran.mean()

            # Overlapping histograms with semi-transparent bars
            ax_hist.hist(ari_opt, bins=40, color='#2A9D8F', alpha=0.4,
                         edgecolor='#2A9D8F', linewidth=0.8,
                         label=f'K={k_opt_label} (mean={mean_ari_opt:.3f})')
            ax_hist.hist(ari_gran, bins=40, color='#E76F51', alpha=0.4,
                         edgecolor='#E76F51', linewidth=0.8,
                         label=f'K={k_gran_label} (mean={mean_ari_gran:.3f})')

            # Mean lines
            ax_hist.axvline(mean_ari_opt, color='#2A9D8F', linewidth=2, linestyle='--')
            ax_hist.axvline(mean_ari_gran, color='#E76F51', linewidth=2, linestyle='--')
            ax_hist.axvline(0, color='gray', linewidth=1, linestyle=':')

            ax_hist.set_xlabel('Adjusted Rand Index', fontsize=11)
            ax_hist.set_ylabel('Count', fontsize=11)
            ax_hist.set_title('Random Gene Sets vs TUS Cluster Similarity', fontsize=12)
            ax_hist.legend(loc='upper right', fontsize=10)
            ax_hist.set_xlim(-0.1, 1.0)
            ax_hist.text(-0.08, 1.05, panel_label, transform=ax_hist.transAxes,
                        fontsize=16, fontweight='bold')

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Figure saved: {output_path}.pdf")

    return fig


# =============================================================================
# FIGURE 2: MONTE CARLO VALIDATION
# =============================================================================

def create_figure2_monte_carlo(monte_carlo_results=None, tus_silhouette=None, output_path=None):
    """
    Create Figure 2: Monte Carlo validation results at K=7.

    Shows ARI distribution comparing TUS gene cluster assignments
    to random gene sets clustered at K=7.

    Args:
        monte_carlo_results: (deprecated) Will load k7 results automatically
        tus_silhouette: (deprecated, kept for compatibility)
        output_path: Base path for output

    Returns:
        fig: Matplotlib figure object
    """
    import pandas as pd

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'supplemental_monte_carlo')

    # Load K=7 results
    mc_k7 = pd.read_csv(os.path.join(config.TABLES_DIR, 'monte_carlo_dual_k_results.csv'))

    k_gran = config.K_GRANULAR
    mean_ari = mc_k7['ari_k7'].mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(mc_k7['ari_k7'], bins=40, color='#888888',
            alpha=0.8, edgecolor='white', linewidth=0.3)

    ax.axvline(mean_ari, color='black', linewidth=2, linestyle='--',
               label=f'Mean ARI = {mean_ari:.2f}')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Adjusted Rand Index (ARI)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(-0.05, 1.0)
    ax.legend(loc='upper right', fontsize=11, frameon=False)

    plt.tight_layout()

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Figure saved: {output_path}.pdf")

    return fig


def create_null_distribution_figure(null_results, output_path=None):
    """
    Create figure comparing TUS-vs-random ARI to null distribution (random-vs-random).

    This figure answers: "Is TUS clustering similarity to random gene sets
    typical, or does TUS show unusual specificity?"

    Panels:
        a) K_OPTIMAL: Null distribution with TUS ARI marked
        b) K_GRANULAR: Null distribution with TUS ARI marked
        c) Summary interpretation

    Args:
        null_results: Dictionary from monte_carlo_null_distribution()
        output_path: Base path for output

    Returns:
        fig: Matplotlib figure object
    """
    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'null_distribution_comparison')

    # Get dynamic K values
    k_opt = config.K_OPTIMAL
    k_gran = config.K_GRANULAR

    fig = plt.figure(figsize=(14, 5))

    # Panel A: K_OPTIMAL null distribution
    ax1 = fig.add_subplot(1, 3, 1)

    null_opt = null_results[f'null_ari_k{k_opt}']
    tus_opt = null_results[f'tus_ari_k{k_opt}']
    pct_opt = null_results[f'tus_percentile_k{k_opt}']

    ax1.hist(null_opt, bins=50, color='#7f8c8d', alpha=0.7, edgecolor='black',
             linewidth=0.5, label='Random vs Random')
    ax1.axvline(tus_opt, color='#e74c3c', linewidth=3, linestyle='-',
                label=f'TUS vs Random: {tus_opt:.3f}')
    ax1.axvline(null_opt.mean(), color='#2c3e50', linewidth=2, linestyle='--',
                label=f'Null mean: {null_opt.mean():.3f}')

    ax1.set_xlabel('Adjusted Rand Index (ARI)', fontsize=11)
    ax1.set_ylabel(f'Count (n={len(null_opt):,} comparisons)', fontsize=11)
    ax1.set_title(f'K={k_opt}: TUS vs Null Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)

    # Add percentile annotation
    if pct_opt > 95:
        color = '#e74c3c'
        text = f'TUS at {pct_opt:.0f}th percentile\n(MORE similar than typical)'
    elif pct_opt < 5:
        color = '#27ae60'
        text = f'TUS at {pct_opt:.0f}th percentile\n(LESS similar = SPECIFIC!)'
    else:
        color = '#3498db'
        text = f'TUS at {pct_opt:.0f}th percentile\n(Typical similarity)'

    ax1.text(0.95, 0.95, text, transform=ax1.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', horizontalalignment='right', color=color,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.text(-0.1, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    # Panel B: K_GRANULAR null distribution
    ax2 = fig.add_subplot(1, 3, 2)

    null_gran = null_results[f'null_ari_k{k_gran}']
    tus_gran = null_results[f'tus_ari_k{k_gran}']
    pct_gran = null_results[f'tus_percentile_k{k_gran}']

    ax2.hist(null_gran, bins=50, color='#7f8c8d', alpha=0.7, edgecolor='black',
             linewidth=0.5, label='Random vs Random')
    ax2.axvline(tus_gran, color='#e74c3c', linewidth=3, linestyle='-',
                label=f'TUS vs Random: {tus_gran:.3f}')
    ax2.axvline(null_gran.mean(), color='#2c3e50', linewidth=2, linestyle='--',
                label=f'Null mean: {null_gran.mean():.3f}')

    ax2.set_xlabel('Adjusted Rand Index (ARI)', fontsize=11)
    ax2.set_ylabel(f'Count (n={len(null_gran):,} comparisons)', fontsize=11)
    ax2.set_title(f'K={k_gran}: TUS vs Null Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)

    # Add percentile annotation
    if pct_gran > 95:
        color = '#e74c3c'
        text = f'TUS at {pct_gran:.0f}th percentile\n(MORE similar than typical)'
    elif pct_gran < 5:
        color = '#27ae60'
        text = f'TUS at {pct_gran:.0f}th percentile\n(LESS similar = SPECIFIC!)'
    else:
        color = '#3498db'
        text = f'TUS at {pct_gran:.0f}th percentile\n(Typical similarity)'

    ax2.text(0.95, 0.95, text, transform=ax2.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', horizontalalignment='right', color=color,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(-0.1, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    # Panel C: Summary interpretation
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')

    summary = f"""
NULL DISTRIBUTION ANALYSIS
{'='*40}

QUESTION:
Is TUS-vs-random similarity typical,
or does TUS show unusual specificity?

METHOD:
• Built null distribution by comparing
  random gene sets to each other
• {len(null_opt):,} random-vs-random comparisons
• Compared TUS-vs-random ARI to null

RESULTS:
K={k_opt}:
  Null (random vs random): {null_opt.mean():.3f} ± {null_opt.std():.3f}
  TUS vs random: {tus_opt:.3f}
  TUS percentile: {pct_opt:.1f}%

K={k_gran}:
  Null (random vs random): {null_gran.mean():.3f} ± {null_gran.std():.3f}
  TUS vs random: {tus_gran:.3f}
  TUS percentile: {pct_gran:.1f}%

INTERPRETATION:
"""

    # Add interpretation
    if pct_opt > 95 and pct_gran > 95:
        summary += "TUS clusters are MORE similar to random\nthan random is to itself → strongly\narchitecture-driven, no TUS specificity"
    elif pct_opt < 5 and pct_gran < 5:
        summary += "TUS clusters are LESS similar to random\nthan random is to itself → TUS genes\nshow SPECIFIC clustering patterns!"
    elif 5 <= pct_opt <= 95 and 5 <= pct_gran <= 95:
        summary += "TUS similarity to random is TYPICAL\n→ TUS behaves like any gene set,\nno special specificity detected"
    else:
        summary += f"Mixed results:\nK={k_opt}: {pct_opt:.0f}th pct, K={k_gran}: {pct_gran:.0f}th pct"

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#cccccc'))

    ax3.text(-0.05, 1.02, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Figure saved: {output_path}.pdf")

    return fig


# =============================================================================
# FIGURE 3: SILHOUETTE CURVE
# =============================================================================

def create_silhouette_curve_figure(k_values, silhouettes, output_path=None):
    """
    Create silhouette score vs K figure.

    Args:
        k_values: Array of K values
        silhouettes: Corresponding silhouette scores
        output_path: Base path for output

    Returns:
        fig: Matplotlib figure object
    """
    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'silhouette_curve')

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, silhouettes, 'o-', color='steelblue', linewidth=2, markersize=8)

    # Highlight K_OPTIMAL and K_GRANULAR
    idx_opt = list(k_values).index(config.K_OPTIMAL) if config.K_OPTIMAL in k_values else None
    idx_gran = list(k_values).index(config.K_GRANULAR) if config.K_GRANULAR in k_values else None

    if idx_opt is not None:
        ax.scatter([k_values[idx_opt]], [silhouettes[idx_opt]], s=200, c='#E64B35',
                   zorder=5, label=f'K={config.K_OPTIMAL} (optimal)', marker='*')
    if idx_gran is not None:
        ax.scatter([k_values[idx_gran]], [silhouettes[idx_gran]], s=150, c='#00A087',
                   zorder=5, label=f'K={config.K_GRANULAR} (granular)', marker='s')

    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Quality vs Number of Clusters')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Figure saved: {output_path}.pdf")

    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE: ROBUST SILHOUETTE ANALYSIS
# =============================================================================

def create_supplemental_robust_silhouette(robust_results_df, output_path=None, n_seeds=None):
    """
    Create supplemental figure showing robust silhouette analysis.

    This figure demonstrates proper statistical methodology for K selection:
    - Silhouette scores averaged across multiple random seeds
    - 95% confidence intervals
    - Local maxima highlighted
    - Excludes K=2 (trivial cortex/subcortex split)

    Args:
        robust_results_df: DataFrame from statistics.compute_robust_silhouette_curve()
                          Can also accept older format with 'mean' and 'std' columns
        output_path: Base path for output
        n_seeds: Number of seeds used (for CI calculation if not in data)

    Returns:
        fig: Matplotlib figure object
    """
    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'supplemental_robust_silhouette')

    # Handle different column name formats
    df = robust_results_df.copy()

    # Map old column names to new ones if needed
    if 'mean' in df.columns and 'mean_silhouette' not in df.columns:
        df['mean_silhouette'] = df['mean']
    if 'std' in df.columns and 'std_silhouette' not in df.columns:
        df['std_silhouette'] = df['std']

    # Calculate CI95 if not present
    if 'ci95' not in df.columns:
        # Default to 1000 seeds if not specified
        seeds = n_seeds if n_seeds is not None else 1000
        df['ci95'] = 1.96 * df['std_silhouette'] / np.sqrt(seeds)

    # Calculate local maxima if not present
    if 'is_local_max' not in df.columns:
        sil_values = df['mean_silhouette'].values
        is_local_max = np.zeros(len(sil_values), dtype=bool)
        for i in range(1, len(sil_values) - 1):
            if sil_values[i] > sil_values[i-1] and sil_values[i] > sil_values[i+1]:
                is_local_max[i] = True
        df['is_local_max'] = is_local_max

    # Filter to K >= 3 (exclude trivial K=2 split)
    plot_df = df[df['k'] >= 3].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = plot_df['k'].values
    mean_sil = plot_df['mean_silhouette'].values
    ci95 = plot_df['ci95'].values

    # Plot with confidence intervals
    ax.fill_between(k_values, mean_sil - ci95, mean_sil + ci95,
                    alpha=0.2, color='black')
    ax.plot(k_values, mean_sil, 'o-', color='black', linewidth=1.5,
            markersize=6)

    # Highlight selected K value
    k_gran = config.K_GRANULAR

    k_gran_idx = np.where(k_values == k_gran)[0]
    if len(k_gran_idx) > 0:
        ax.scatter([k_gran], [mean_sil[k_gran_idx[0]]], s=120, c='magenta', zorder=6,
                   marker='o')

    # Set y-axis limits with some padding
    y_min = (mean_sil - ci95).min()
    y_max = (mean_sil + ci95).max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Number of Clusters (K)', fontsize=14)
    ax.set_ylabel('Silhouette Score', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticks(k_values)

    plt.tight_layout()

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Supplemental figure saved: {output_path}.pdf")

    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE: MONTE CARLO EXPLANATION
# =============================================================================

def create_monte_carlo_explanation_figure(
    monte_carlo_dual_results,
    tus_silhouette_opt,
    tus_silhouette_gran,
    output_path=None
):
    """
    Create comprehensive Monte Carlo validation figure.

    This figure explains whether TUS gene clusters are specific to mechanosensitive
    genes or simply reflect general brain architecture.

    Four panels:
    a) ARI distribution at K_OPTIMAL - Are random clusters similar to TUS K_OPTIMAL?
    b) ARI distribution at K_GRANULAR - Are random clusters similar to TUS K_GRANULAR?
    c) Silhouette comparison - Do TUS genes cluster better/worse than random?
    d) Interpretation summary - What does this mean?

    Args:
        monte_carlo_dual_results: DataFrame from monte_carlo_dual_k_validation()
        tus_silhouette_opt: TUS silhouette at K_OPTIMAL
        tus_silhouette_gran: TUS silhouette at K_GRANULAR
        output_path: Base path for output

    Returns:
        fig: Matplotlib figure object
    """
    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'supplemental_monte_carlo')

    # Get K=7 ARI column
    k_gran = config.K_GRANULAR
    ari_col = f'ari_k{k_gran}'
    ari_vals = monte_carlo_dual_results[ari_col].values
    mean_ari = ari_vals.mean()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(ari_vals, bins=50, color='#888888', alpha=0.8,
            edgecolor='white', linewidth=0.3)

    ax.axvline(mean_ari, color='black', linewidth=2, linestyle='--',
               label=f'Mean ARI = {mean_ari:.2f}')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Adjusted Rand Index (ARI)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(-0.05, 1.0)
    ax.legend(loc='upper right', fontsize=11, frameon=False)

    plt.tight_layout()

    # Save
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.jpg', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Monte Carlo explanation figure saved: {output_path}.pdf")

    return fig


# =============================================================================
# GENE-CLUSTER ASSOCIATION FIGURES
# =============================================================================

def create_gene_cluster_bubble_plot(cluster_means, genes, labels, output_path=None):
    """
    Create bubble plot showing gene expression profiles across clusters.

    Bubble size = |z-score|, color = expression direction (red=high, blue=low).
    Includes gene family color bar and cluster type indicators.

    Args:
        cluster_means: Array (n_clusters x n_genes) of mean z-scores
        genes: List of gene names
        labels: Cluster assignments for determining cluster types
        output_path: Base path for output

    Returns:
        fig: Matplotlib figure object
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from scipy.cluster.hierarchy import linkage, leaves_list

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'gene_cluster_bubble_plot')

    n_clusters = cluster_means.shape[0]
    n_genes = len(genes)

    # Determine cluster types
    n_cortical = config.N_CORTICAL_REGIONS
    cluster_types = []
    for c in range(n_clusters):
        mask = labels == c
        n_cort = (mask[:n_cortical]).sum()
        n_sub = (mask[n_cortical:]).sum()
        if n_sub == 0:
            cluster_types.append('Cortical')
        elif n_cort == 0:
            cluster_types.append('Subcortical')
        else:
            cluster_types.append('Mixed')

    # Hierarchical clustering for ordering
    gene_linkage = linkage(cluster_means.T, method='average', metric='euclidean')
    gene_order = leaves_list(gene_linkage)
    cluster_linkage = linkage(cluster_means, method='average', metric='euclidean')
    cluster_order = leaves_list(cluster_linkage)

    ordered_genes = [genes[i] for i in gene_order]
    ordered_cluster_types = [cluster_types[i] for i in cluster_order]
    ordered_cluster_labels = [f'C{i+1}' for i in cluster_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#2166AC', '#4393C3', '#92C5DE', '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
    )

    # Plot bubbles
    for i, cluster_idx in enumerate(cluster_order):
        for j, gene_idx in enumerate(gene_order):
            val = cluster_means[cluster_idx, gene_idx]
            size = (abs(val) / 4.0) * 600
            color = cmap((val + 4) / 8)
            circle = plt.Circle((j, i), np.sqrt(size/np.pi)/12,
                               color=color, ec='white', linewidth=0.5)
            ax.add_patch(circle)

    # Gene category color bar on top
    for j, gene_idx in enumerate(gene_order):
        gene = genes[gene_idx]
        cat = config.GENE_CATEGORIES.get(gene, 'Other')
        rect = FancyBboxPatch((j-0.4, n_clusters+0.2), 0.8, 0.4,
                              boxstyle='round,pad=0.02',
                              facecolor=config.CATEGORY_COLORS.get(cat, '#999999'),
                              edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

    # Cluster type indicator
    type_colors = {'Cortical': '#3A86FF', 'Subcortical': '#FF006E', 'Mixed': '#8338EC'}
    for i, ctype in enumerate(ordered_cluster_types):
        rect = FancyBboxPatch((-1.5, i-0.3), 0.6, 0.6,
                              boxstyle='round,pad=0.02',
                              facecolor=type_colors[ctype],
                              edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

    ax.set_xlim(-2, n_genes)
    ax.set_ylim(-0.8, n_clusters + 1)
    ax.set_xticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(ordered_cluster_labels, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(-4, 4))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label('Mean Z-score', fontsize=11)

    # Legends
    cat_patches = [Patch(facecolor=config.CATEGORY_COLORS[cat], label=cat, edgecolor='white')
                   for cat in config.CATEGORY_ORDER]
    leg1 = ax.legend(handles=cat_patches, loc='upper left', bbox_to_anchor=(1.12, 1.0),
                     title='Gene Family', frameon=True, fontsize=9)
    ax.add_artist(leg1)

    type_patches = [Patch(facecolor=type_colors[t], label=t, edgecolor='white')
                    for t in ['Cortical', 'Subcortical', 'Mixed']]
    ax.legend(handles=type_patches, loc='upper left', bbox_to_anchor=(1.12, 0.55),
              title='Cluster Type', frameon=True, fontsize=9)

    ax.set_title('Gene Expression Profiles Across Spatial Clusters',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.png', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Gene-cluster bubble plot saved: {output_path}.pdf")

    return fig


def create_gene_cluster_network(cluster_means, genes, labels, output_path=None, threshold=1.0):
    """
    Create network diagram showing gene-cluster associations.

    Genes on left, clusters on right, connected by lines weighted by expression.
    Line thickness and color indicate association strength and direction.

    Args:
        cluster_means: Array (n_clusters x n_genes) of mean z-scores
        genes: List of gene names
        labels: Cluster assignments
        output_path: Base path for output
        threshold: Minimum |z-score| to show connection

    Returns:
        fig: Matplotlib figure object
    """
    from matplotlib.colors import LinearSegmentedColormap

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'gene_cluster_network')

    n_clusters = cluster_means.shape[0]
    n_genes = len(genes)

    # Determine cluster types and sizes
    n_cortical = config.N_CORTICAL_REGIONS
    cluster_types = []
    cluster_sizes = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_sizes.append(mask.sum())
        n_cort = (mask[:n_cortical]).sum()
        n_sub = (mask[n_cortical:]).sum()
        if n_sub == 0:
            cluster_types.append('Cortical')
        elif n_cort == 0:
            cluster_types.append('Subcortical')
        else:
            cluster_types.append('Mixed')

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Positions
    gene_y = np.linspace(0.95, 0.05, n_genes)
    gene_x = np.zeros(n_genes) + 0.15
    cluster_y = np.linspace(0.85, 0.15, n_clusters)
    cluster_x = np.zeros(n_clusters) + 0.85

    # Colormaps for connections
    cmap_pos = LinearSegmentedColormap.from_list('pos', ['#FDDBC7', '#F4A582', '#D6604D', '#B2182B'])
    cmap_neg = LinearSegmentedColormap.from_list('neg', ['#D1E5F0', '#67A9CF', '#2166AC', '#053061'])

    # Draw connections
    for g_idx, gene in enumerate(genes):
        for c_idx in range(n_clusters):
            val = cluster_means[c_idx, g_idx]
            if abs(val) > threshold:
                alpha = min(abs(val) / 3.5, 1.0) * 0.7
                lw = abs(val) / 3.5 * 3 + 0.5
                if val > 0:
                    color = cmap_pos(min(val/3.5, 1.0))
                else:
                    color = cmap_neg(min(abs(val)/3.5, 1.0))
                ax.plot([gene_x[g_idx], cluster_x[c_idx]],
                       [gene_y[g_idx], cluster_y[c_idx]],
                       color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Draw gene nodes
    for g_idx, gene in enumerate(genes):
        cat = config.GENE_CATEGORIES.get(gene, 'Other')
        color = config.CATEGORY_COLORS.get(cat, '#999999')
        circle = plt.Circle((gene_x[g_idx], gene_y[g_idx]), 0.018,
                           facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(gene_x[g_idx] - 0.03, gene_y[g_idx], gene, ha='right', va='center',
               fontsize=9, fontweight='bold', color=color)

    # Draw cluster nodes
    type_colors = {'Cortical': '#3A86FF', 'Subcortical': '#FF006E', 'Mixed': '#8338EC'}
    for c_idx in range(n_clusters):
        size = 0.02 + cluster_sizes[c_idx] / 300 * 0.03
        circle = plt.Circle((cluster_x[c_idx], cluster_y[c_idx]), size,
                           facecolor=type_colors[cluster_types[c_idx]],
                           edgecolor='white', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(cluster_x[c_idx] + 0.05, cluster_y[c_idx],
               f'C{c_idx+1} (n={cluster_sizes[c_idx]})',
               ha='left', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax.text(0.15, 1.02, 'TUS Genes', ha='center', fontsize=14, fontweight='bold')
    ax.text(0.85, 1.02, 'Spatial Clusters', ha='center', fontsize=14, fontweight='bold')

    # Legends
    cat_patches = [Patch(facecolor=config.CATEGORY_COLORS[cat], label=cat, edgecolor='white')
                   for cat in config.CATEGORY_ORDER]
    leg1 = ax.legend(handles=cat_patches, loc='lower left', bbox_to_anchor=(0.0, 0.0),
                     title='Gene Family', frameon=True, fontsize=9, ncol=3)
    ax.add_artist(leg1)

    type_patches = [Patch(facecolor=type_colors[t], label=t, edgecolor='white')
                    for t in ['Cortical', 'Subcortical', 'Mixed']]
    ax.legend(handles=type_patches, loc='lower right', bbox_to_anchor=(1.0, 0.0),
              title='Cluster Type', frameon=True, fontsize=9)

    ax.text(0.5, -0.08, f'Line thickness and color intensity = |z-score|; Red = high, Blue = low (threshold = {threshold})',
            ha='center', fontsize=10, style='italic', transform=ax.transAxes)

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('Gene-Cluster Expression Network', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path + '.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + '.png', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Gene-cluster network saved: {output_path}.pdf")

    return fig


# =============================================================================
# FIGURE 1 MAIN: K_GRANULAR CLUSTERING + GENE PROFILES (COMBINED)
# =============================================================================

def create_figure1_main(
    cluster_results,
    cluster_gene_matrix,
    gene_linkage,
    cluster_linkage_9,
    atlas_img,
    atlas_data,
    genes,
    gene_significance=None,
    output_path=None
):
    """
    Create Figure 1 Main: K_GRANULAR clustering overview with gene expression profiles.

    Combined figure with 6 rows:
      Row a: K_GRANULAR heatmap with hierarchical gene ordering
      Row b: 3D brain render (PyVista) - each cluster highlighted
      Row c: Glass brain projection (nilearn) - axial view
      Row d: Axial slice at cluster's peak location
      Row e: Pattern radar plot (z-score) with significance markers
      Row f: Magnitude radar plot (vs brain mean)

    Args:
        cluster_results: Dict from clustering.get_cluster_results()
        cluster_gene_matrix: DataFrame with cluster means per gene (rows=clusters, cols=genes)
        gene_linkage: Hierarchical linkage for genes
        cluster_linkage_9: Hierarchical linkage for K_GRANULAR clusters
        atlas_img: Nibabel image object for the brain atlas
        atlas_data: 3D numpy array of atlas parcellation
        genes: List of gene names
        gene_significance: Dict from clustering.compute_gene_significance() (optional)
        output_path: Base path for output (default: output/figures/figure1_main)

    Returns:
        fig: Matplotlib figure object
    """
    import pandas as pd
    import pyvista as pv
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from matplotlib.transforms import blended_transform_factory
    from nilearn import plotting
    from nilearn.image import new_img_like
    from scipy.cluster.hierarchy import dendrogram

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'figure1_main')

    # Enable off-screen rendering for PyVista
    pv.OFF_SCREEN = True

    n_clusters = config.K_GRANULAR
    n_genes = len(genes)

    # Get cluster data
    labels_granular = cluster_results[config.K_GRANULAR]['labels']
    labels_opt = cluster_results[config.K_OPTIMAL]['labels']
    cluster_means_granular = cluster_results[config.K_GRANULAR]['cluster_means']
    sil_granular = cluster_results[config.K_GRANULAR]['silhouette']

    # Compute K_GRANULAR to K_OPTIMAL parent mapping
    k_granular_to_k_opt_map = {}
    for c in range(config.K_GRANULAR):
        regions_in_c = np.where(labels_granular == c)[0]
        k_opt_counts = np.bincount(labels_opt[regions_in_c], minlength=config.K_OPTIMAL)
        k_granular_to_k_opt_map[c] = int(np.argmax(k_opt_counts))

    # Generate K_GRANULAR colors as shades of their K_OPTIMAL parent colors
    gran_colors = generate_granular_colors_from_parent_map(k_granular_to_k_opt_map, config.CLUSTER_COLORS_OPT)
    gran_colors[1], gran_colors[4] = gran_colors[4], gran_colors[1]  # Swap C2 <-> C5

    # Build parcel mapping from atlas
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude background
    parcel_to_cluster = {}
    for i, parcel in enumerate(sorted(unique_parcels)):
        if i < len(labels_granular):
            parcel_to_cluster[int(parcel)] = labels_granular[i] + 1  # 1-indexed

    # Find best axial slice for each cluster
    best_slices = {}
    for c in range(1, n_clusters + 1):
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        if cluster_parcels:
            cluster_mask = np.isin(atlas_data, cluster_parcels)
            slice_areas = [np.sum(cluster_mask[:, :, z]) for z in range(atlas_data.shape[2])]
            best_slices[c] = np.argmax(slice_areas)
        else:
            best_slices[c] = atlas_data.shape[2] // 2

    # Create figure: 5 content rows + 1 spacer (row f magnitude radar disabled)
    # Row e: Pattern (z-score) radar
    fig = plt.figure(figsize=(18, 16))

    # Panel label styling (consistent across all rows)
    LABEL_X = 0.005          # Fixed x in figure coords (left edge)
    LABEL_PROPS = dict(fontsize=36, fontweight='normal', fontfamily='Arial',
                       color=(30/255, 30/255, 30/255), ha='left', va='bottom')

    def add_panel_label(ax, letter, y=1.05):
        """Place a panel label at a fixed left-edge x position (figure coords)."""
        trans = blended_transform_factory(fig.transFigure, ax.transAxes)
        ax.text(LABEL_X, y, letter, transform=trans, **LABEL_PROPS)

    # GridSpec for the main layout (spacer row at index 1 pushes panel B down ~1cm)
    gs_main = GridSpec(6, 1, figure=fig, height_ratios=[1.2, 0.12, 0.8, 0.8, 0.8, 1.0],
                       left=0.01, right=0.99, top=0.95, bottom=0.03, hspace=0.15)

    # Row a: Heatmap (spanning full width)
    gs_heatmap = gs_main[0].subgridspec(1, 1)
    ax_heatmap_outer = fig.add_subplot(gs_heatmap[0])
    ax_heatmap_outer.axis('off')

    # Create heatmap with dendrograms
    gs_heatmap_inner = gs_heatmap[0].subgridspec(
        2, 3,
        height_ratios=[0.15, 1],
        width_ratios=[0.04, 1, 0.015],
        hspace=0.02, wspace=0.01
    )

    ax_row_dend = fig.add_subplot(gs_heatmap_inner[1, 0])
    ax_col_dend = fig.add_subplot(gs_heatmap_inner[0, 1])
    ax_heatmap = fig.add_subplot(gs_heatmap_inner[1, 1])
    ax_colorbar = fig.add_subplot(gs_heatmap_inner[1, 2])

    # Column dendrogram (genes)
    dend_col = dendrogram(
        gene_linkage,
        labels=genes,
        orientation='top',
        leaf_font_size=0,
        color_threshold=0,
        above_threshold_color=config.LINE_COLOR,
        ax=ax_col_dend
    )
    ax_col_dend.axis('off')
    add_panel_label(ax_col_dend, 'a', y=1.2)
    ax_col_dend.set_title(f'K={config.K_GRANULAR} Gene Expression (Silhouette={sil_granular:.3f})',
                          fontsize=11, pad=10)

    gene_order = [genes[i] for i in dend_col['leaves']]
    gene_indices = [genes.index(g) for g in gene_order]

    # Row dendrogram (clusters) - draw with colored branches
    # Use dendrogram with custom leaf order matching row_order
    # Custom row order: C3, C1, C7, C5, C4, C2, C6 (0-indexed: 2, 0, 6, 4, 3, 1, 5)
    row_order = [2, 0, 6, 4, 3, 1, 5]

    # We need to reorder the linkage to match our custom order
    # Use dendrogram's no_plot to get structure, then draw manually
    dend_row = dendrogram(
        cluster_linkage_9,
        orientation='left',
        leaf_font_size=0,
        color_threshold=0,
        above_threshold_color=config.LINE_COLOR,
        ax=ax_row_dend,
        no_plot=True
    )

    # The dendrogram has its own leaf order. We need to map its y-positions
    # to our row_order. Both have 7 leaves at y = 5, 15, 25, 35, 45, 55, 65
    dend_leaves = dend_row['leaves']  # e.g. [0, 1, 3, 5, 2, 4, 6]

    # Build mapping: for each cluster index, what's its y-position in dendrogram vs our order
    dend_cluster_to_y = {leaf: i * 10 + 5 for i, leaf in enumerate(dend_leaves)}
    our_cluster_to_y = {leaf: i * 10 + 5 for i, leaf in enumerate(row_order)}

    # Remap dendrogram segments from dend leaf positions to our row_order positions
    dend_y_to_our_y = {}
    for cluster in range(n_clusters):
        dend_y_to_our_y[dend_cluster_to_y[cluster]] = our_cluster_to_y[cluster]

    # Also need to remap internal node positions
    # Process segments to find internal node y-positions and remap them
    # Internal nodes are at midpoints - we need to recursively remap
    def remap_y(y):
        if y in dend_y_to_our_y:
            return dend_y_to_our_y[y]
        return y  # Will be filled in during processing

    # First pass: build internal node mapping by processing linkage
    # Each segment connects two children at y1, y2 with internal node at (y1+y2)/2
    # We need to remap children first, then compute new midpoints
    for ic, dc in zip(dend_row['icoord'], dend_row['dcoord']):
        y1_old, _, y2_old, _ = ic
        y1_new = remap_y(y1_old)
        y2_new = remap_y(y2_old)
        mid_old = (y1_old + y2_old) / 2
        mid_new = (y1_new + y2_new) / 2
        dend_y_to_our_y[mid_old] = mid_new

    # Map leaf y-positions (in our order) to cluster colors
    leaf_y_to_color = {our_cluster_to_y[c]: gran_colors[c] for c in range(n_clusters)}

    def get_color_for_y(y):
        """Find the color for a given y position by nearest leaf."""
        if y in leaf_y_to_color:
            return leaf_y_to_color[y]
        closest_y = min(leaf_y_to_color.keys(), key=lambda ly: abs(ly - y))
        return leaf_y_to_color[closest_y]

    # Draw each segment with colored branches (remapped to our row order)
    # Track leaf branch extents for label placement
    leaf_branch_extent = {}  # leaf_y -> (x_start, x_end)
    near_black = (25/255, 25/255, 25/255)

    for ic, dc in zip(dend_row['icoord'], dend_row['dcoord']):
        y1_old, _, y2_old, _ = ic
        x_left1, x_top, _, x_left2 = dc

        y1 = dend_y_to_our_y.get(y1_old, y1_old)
        y2 = dend_y_to_our_y.get(y2_old, y2_old)

        color1 = get_color_for_y(y1)
        color2 = get_color_for_y(y2)

        # All branches near-black
        ax_row_dend.plot([x_left1, x_top], [y1, y1],
                         color=near_black, linewidth=1.5, zorder=2)
        ax_row_dend.plot([x_left2, x_top], [y2, y2],
                         color=near_black, linewidth=1.5, zorder=2)
        # Vertical connecting bar
        ax_row_dend.plot([x_top, x_top], [y1, y2], color=near_black, linewidth=1.5, zorder=1)

        # Track leaf branches (those starting at x=0)
        if abs(x_left1) < 0.01:
            leaf_branch_extent[y1] = (x_left1, x_top)
        if abs(x_left2) < 0.01:
            leaf_branch_extent[y2] = (x_left2, x_top)

    # Set axis limits - extend left margin to accommodate circles + labels on the right
    max_x = max(max(dc) for dc in dend_row['dcoord'])
    ax_row_dend.set_xlim(max_x * 1.05, -max_x * 0.65)
    ax_row_dend.set_ylim(-0.5, n_clusters * 10 + 0.5)

    # Add filled circles with gap from branch ends, labels above branch lines
    circle_x = -max_x * 0.18  # circle to the right of branch end (x=0), with gap
    for i, c in enumerate(row_order):
        leaf_y = i * 10 + 5
        color = gran_colors[c]
        # Circle to the right of the branch end
        ax_row_dend.plot(circle_x, leaf_y, 'o', color=color, markersize=9,
                         markeredgecolor='none', zorder=5, clip_on=False)
        # Label above the branch line, to the left of the circle
        if leaf_y in leaf_branch_extent:
            x_start, x_end = leaf_branch_extent[leaf_y]
            label_x = (x_start + x_end) / 2  # centered on branch line
        else:
            label_x = max_x * 0.3
        ax_row_dend.text(label_x, leaf_y, f'C{c+1}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold',
                         color=color, zorder=5, clip_on=False)

    ax_row_dend.axis('off')
    ax_row_dend.invert_yaxis()

    # Heatmap
    heatmap_data = cluster_means_granular[row_order, :][:, gene_indices]
    im = ax_heatmap.imshow(
        heatmap_data, aspect='auto', cmap=HEATMAP_CMAP, vmin=-3, vmax=3
    )

    ax_heatmap.set_yticks([])
    ax_heatmap.set_yticklabels([])
    ax_heatmap.set_xticks(range(len(gene_order)))
    ax_heatmap.set_xticklabels(gene_order, rotation=45, fontsize=8,
                                ha='right', rotation_mode='anchor')
    ax_heatmap.tick_params(axis='x', length=0)
    plt.colorbar(im, cax=ax_colorbar, label='Z-score')

    # Get affine transform for proper coordinates
    affine = atlas_img.affine
    spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
    origin = affine[:3, 3]

    # Custom ordering for rows b-f: cortical (broad→narrow) then subcortical
    # Order: green(C3), gold(C1), lightblue(C7), darkred(C5), pink(C4), yellow(C2), lightred(C6)
    radar_row_order = [2, 0, 6, 4, 3, 1, 5]

    # Row b: 3D brain renderings (matching radar row order)
    gs_3d = gs_main[2].subgridspec(1, n_clusters, wspace=0.02)
    print('Generating 3D brain renderings...')
    for col_idx, cluster_idx in enumerate(radar_row_order):
        c = cluster_idx + 1  # Convert 0-indexed to 1-indexed cluster number
        ax = fig.add_subplot(gs_3d[0, col_idx])
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_mask = np.isin(atlas_data, cluster_parcels).astype(float)
        other_mask = np.isin(atlas_data, other_parcels).astype(float)

        plotter = pv.Plotter(off_screen=True, window_size=[500, 500])
        plotter.set_background('white')

        if np.any(other_mask):
            try:
                other_grid = pv.ImageData(dimensions=other_mask.shape, spacing=spacing, origin=origin)
                other_grid.point_data['values'] = other_mask.flatten(order='F')
                other_surface = other_grid.contour([0.5], scalars='values')
                if other_surface.n_points > 0:
                    plotter.add_mesh(other_surface, color='gray', opacity=0.15, smooth_shading=True)
            except Exception:
                pass

        cluster_color = gran_colors[cluster_idx]
        if np.any(cluster_mask):
            try:
                cluster_grid = pv.ImageData(dimensions=cluster_mask.shape, spacing=spacing, origin=origin)
                cluster_grid.point_data['values'] = cluster_mask.flatten(order='F')
                cluster_surface = cluster_grid.contour([0.5], scalars='values')
                if cluster_surface.n_points > 0:
                    plotter.add_mesh(cluster_surface, color=cluster_color, opacity=0.5, smooth_shading=True)
            except Exception:
                pass

        plotter.camera.position = (563.2, 446.6, 158.8)
        plotter.camera.focal_point = (180.0, -17.0, 15.0)
        plotter.camera.up = (-0.15, -0.18, 0.97)
        plotter.camera.zoom(1.85)

        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax.imshow(img)
        if col_idx == 0:
            ax.set_title(f'C{c}', fontsize=10, fontweight='bold')
            add_panel_label(ax, 'b')
        else:
            ax.set_title(f'C{c}', fontsize=10, fontweight='bold')
        ax.axis('off')

    # Row c: Glass brain views (matching radar row order)
    gs_glass = gs_main[3].subgridspec(1, n_clusters, wspace=0.02)
    print('Generating glass brain views...')
    for col_idx, cluster_idx in enumerate(radar_row_order):
        c = cluster_idx + 1  # Convert 0-indexed to 1-indexed cluster number
        ax = fig.add_subplot(gs_glass[0, col_idx])
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_vol = np.zeros_like(atlas_data)
        other_vol = np.zeros_like(atlas_data)

        for p in cluster_parcels:
            cluster_vol[atlas_data == p] = 1
        for p in other_parcels:
            other_vol[atlas_data == p] = 0.3

        cluster_nii = new_img_like(atlas_img, cluster_vol)
        other_nii = new_img_like(atlas_img, other_vol)

        cluster_color = gran_colors[cluster_idx]
        plotting.plot_glass_brain(other_nii, axes=ax, display_mode='z',
                                  cmap='gray', alpha=0.3, colorbar=False,
                                  plot_abs=False, threshold=0.1)
        # Create colormap with built-in transparency
        rgb = mcolors.to_rgb(cluster_color)
        transparent = (rgb[0], rgb[1], rgb[2], 0.0)  # Fully transparent
        semi_opaque = (rgb[0], rgb[1], rgb[2], 0.5)  # Semi-transparent cluster color
        cluster_cmap = mcolors.LinearSegmentedColormap.from_list(f'c{c}', [transparent, semi_opaque])
        plotting.plot_glass_brain(cluster_nii, axes=ax, display_mode='z',
                                  cmap=cluster_cmap, alpha=1.0, colorbar=False,
                                  plot_abs=False, threshold=0.5)
        if col_idx == 0:
            add_panel_label(ax, 'c')
        ax.axis('off')

    # Row d: Axial brain slices (ordered by dendrogram similarity)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    cluster_grays = {c: 0.3 + 0.4 * (c - 1) / (n_clusters - 1) for c in range(1, n_clusters + 1)}

    gs_axial = gs_main[4].subgridspec(1, n_clusters, wspace=0.02)
    for col_idx, cluster_idx in enumerate(radar_row_order):
        c = cluster_idx + 1  # Convert 0-indexed to 1-indexed cluster number
        ax = fig.add_subplot(gs_axial[0, col_idx])
        z = best_slices[c]
        slice_2d = atlas_data[:, :, z]

        rgb_img = np.ones((*slice_2d.shape, 3))
        cluster_rgb = hex_to_rgb(gran_colors[cluster_idx])

        for parcel_id in np.unique(slice_2d):
            if parcel_id == 0:
                continue
            mask = slice_2d == parcel_id
            parcel_cluster = parcel_to_cluster.get(int(parcel_id), 0)
            if parcel_cluster == c:
                rgb_img[mask] = cluster_rgb
            elif parcel_cluster > 0:
                gray_val = cluster_grays[parcel_cluster]
                rgb_img[mask] = [gray_val, gray_val, gray_val]

        rgb_uint8 = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
        ax.imshow(np.rot90(rgb_uint8), aspect='equal')
        ax.set_facecolor('none')
        if col_idx == 0:
            add_panel_label(ax, 'd')
        ax.axis('off')

    # Row e & f: Dual Radar plots - Pattern (z-score) + Magnitude (vs brain mean)
    # Gene labels: a-z (26) + Greek letters (20) = 46 total
    gene_letters = list('abcdefghijklmnopqrstuvwxyz') + [
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ',
        'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ'
    ]

    # Sort genes by category so adjacent genes are from the same family
    genes_by_category = []
    for category in config.CATEGORY_ORDER:
        family_genes = config.GENE_FAMILIES.get(category, [])
        # Only include genes that are in our actual gene list
        for g in family_genes:
            if g in genes:
                genes_by_category.append(g)

    # Add any genes not in a category (shouldn't happen, but safety check)
    for g in genes:
        if g not in genes_by_category:
            genes_by_category.append(g)

    # Map genes to letters (alphabetically assigned to category-ordered genes)
    gene_to_letter = {gene: gene_letters[i] for i, gene in enumerate(genes_by_category)}

    # Gene indices in category order (for extracting from cluster_means_raw)
    gene_indices_cat_order = [genes.index(g) for g in genes_by_category]

    # Compute raw cluster means and brain mean for magnitude calculation
    # cluster_means_granular is z-scored; we need raw expression
    # Get raw expression from cluster_results
    cluster_means_raw = cluster_results[config.K_GRANULAR].get('cluster_means_raw', None)
    if cluster_means_raw is None:
        # Fallback: approximate from z-scores (won't be perfect, but close)
        # Better approach: recalculate from expression_matrix if available
        cluster_means_raw = cluster_means_granular  # Use z-scored as fallback
        brain_mean_per_gene = np.zeros(n_genes)
        cluster_ratio_to_brain = np.ones((n_clusters, n_genes))
    else:
        # Whole-brain mean for each gene
        brain_mean_per_gene = cluster_means_raw.mean(axis=0)
        # Avoid division by zero
        brain_mean_safe = np.where(brain_mean_per_gene > 0, brain_mean_per_gene, 1e-10)
        cluster_ratio_to_brain = cluster_means_raw / brain_mean_safe

    gs_radar_pattern = gs_main[5].subgridspec(1, n_clusters, wspace=0.02)

    # Scaling functions
    def apply_zscore_scaling(z, zero_r=1.5, scale=1.2):
        """For z-scores: tanh compression, center at 1.5"""
        return zero_r + np.tanh(z) * scale

    def apply_ratio_scaling(ratio, one_r=1.5, scale=1.8):
        """For ratios: log scale, center at 1.5 (ratio=1.0)"""
        # log2 scaling: ratio of 2x = +1, ratio of 0.5x = -1
        log_ratio = np.log2(np.clip(ratio, 0.25, 4.0))  # Clip to avoid extremes
        return one_r + log_ratio * scale

    # Each gene gets its own wedge/slice
    wedge_angle = 360.0 / n_genes  # degrees per gene
    angles = np.linspace(0, -360, n_genes, endpoint=False)  # starting angle for each wedge (clockwise)

    # Gene family colors for radar points (now in category order)
    gene_colors = [config.CATEGORY_COLORS.get(config.GENE_CATEGORIES.get(g, 'Other'), '#999999')
                   for g in genes_by_category]

    from matplotlib.patches import Wedge

    # Row e: Pattern (z-scored) radar plots
    for col_idx, cluster_idx in enumerate(radar_row_order):
        c = cluster_idx + 1  # Convert 0-indexed to 1-indexed cluster number
        ax = fig.add_subplot(gs_radar_pattern[0, col_idx])
        ax.set_aspect('equal')
        ax.set_xlim(-3.11, 3.11)
        ax.set_ylim(-3.11, 3.11)
        ax.axis('off')

        # Get z-scored values for this cluster from cluster_gene_matrix (in category order)
        row_name = f'C{c}' if f'C{c}' in cluster_gene_matrix.index else cluster_gene_matrix.index[c-1]
        values = [cluster_gene_matrix.loc[row_name, g] if g in cluster_gene_matrix.columns else 0
                  for g in genes_by_category]

        radii = [max(0.1, min(2.7, apply_zscore_scaling(v))) for v in values]

        # z=0 reference circle
        circle = Circle((0, 0), 1.5, fill=False, edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.plot(0, 0, 'ko', markersize=2, zorder=11)

        # Draw pie slices (wedges) for each gene - NOT connected to neighbors
        for i, gene in enumerate(genes_by_category):
            start_angle = angles[i] - wedge_angle / 2  # Center wedge on the gene's angle
            wedge = Wedge(
                center=(0, 0),
                r=radii[i],
                theta1=start_angle,
                theta2=start_angle + wedge_angle,
                facecolor=gene_colors[i],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(wedge)

        # Gene labels
        for i, gene in enumerate(genes_by_category):
            # Position label at the center angle of each wedge
            angle_rad = np.radians(angles[i])
            x, y = 3.3 * np.cos(angle_rad), 3.3 * np.sin(angle_rad)
            letter = gene_to_letter[gene]

            if col_idx == 0:
                # First column: show gene name only with radial rotation
                rotation = angles[i]  # Use angle directly for radial alignment
                # Flip text that would appear upside down (for negative/clockwise angles)
                if -270 < rotation < -90:
                    rotation += 180
                ax.text(x, y, gene, ha='center', va='center', fontsize=9.75,
                        rotation=rotation, color=gene_colors[i], fontweight='bold')

        if col_idx == 0:
            add_panel_label(ax, 'e')
            ax.text(-0.2, 0.5, 'Pattern\n(z-score)', transform=ax.transAxes,
                    fontsize=9, ha='right', va='center', fontweight='bold')

    # Row f: Magnitude (ratio to brain mean) radar plots — DISABLED
    # To re-enable: set SHOW_ROW_F = True and restore gs_main to 7 rows with
    # height_ratios=[1.2, 0.12, 0.8, 0.8, 0.8, 1.0, 1.0], add gs_radar_magnitude = gs_main[6]
    SHOW_ROW_F = False
    if SHOW_ROW_F:
        gs_radar_magnitude = gs_main[6].subgridspec(1, n_clusters, wspace=0.02)
        for col_idx, cluster_idx in enumerate(radar_row_order):
            c = cluster_idx + 1
            ax = fig.add_subplot(gs_radar_magnitude[0, col_idx])
            ax.set_aspect('equal')
            ax.set_xlim(-3.11, 3.11)
            ax.set_ylim(-3.11, 3.11)
            ax.axis('off')

            ratio_values = cluster_ratio_to_brain[cluster_idx, gene_indices_cat_order]
            radii = [max(0.1, min(2.7, apply_ratio_scaling(v))) for v in ratio_values]

            circle = Circle((0, 0), 1.5, fill=False, edgecolor='black', linewidth=1.5, zorder=10)
            ax.add_patch(circle)
            ax.plot(0, 0, 'ko', markersize=2, zorder=11)

            for i, gene in enumerate(genes_by_category):
                start_angle = angles[i] - wedge_angle / 2
                wedge = Wedge(
                    center=(0, 0),
                    r=radii[i],
                    theta1=start_angle,
                    theta2=start_angle + wedge_angle,
                    facecolor=gene_colors[i],
                    edgecolor='white',
                    linewidth=0.5,
                    alpha=0.7
                )
                ax.add_patch(wedge)

            mean_ratio = ratio_values.mean()
            ax.text(0, -2.55, f'{mean_ratio:.2f}x', ha='center', va='top',
                    fontsize=9, fontweight='bold')

            if col_idx == 0:
                ax.text(-0.15, 1.05, 'f', transform=ax.transAxes, fontsize=32, fontweight='bold', fontfamily='Arial')
                ax.text(-0.2, 0.5, 'Magnitude\n(vs brain)', transform=ax.transAxes,
                        fontsize=9, ha='right', va='center', fontweight='bold')

    # Add legend at bottom
    fig.text(0.5, 0.012, 'Row e: Gene expression PATTERN (z-score, black circle = mean)',
             ha='center', fontsize=9, style='italic')

    # Save
    plt.savefig(output_path + '.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path + '.svg', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path + '.jpg', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path + '.tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure 1 Main saved: {output_path}.pdf/.svg/.tiff/.png')

    return fig


# =============================================================================
# SUPPLEMENTAL FIGURE 1: K_OPTIMAL CLUSTERING
# =============================================================================

def create_supplemental1_optimal(
    cluster_results,
    cluster_gene_matrix_opt,
    gene_linkage,
    atlas_img,
    atlas_data,
    genes,
    output_path=None
):
    """
    Create Supplemental Figure 1: K_OPTIMAL clustering overview.

    Similar layout to Figure 1 Main but for K_OPTIMAL optimal clustering:
      Row a: K_OPTIMAL heatmap with hierarchical gene ordering
      Row b: 3D brain render (PyVista) - each cluster highlighted
      Row c: Glass brain projection (nilearn) - axial view
      Row d: Axial slice at cluster's peak location

    Args:
        cluster_results: Dict from clustering.get_cluster_results()
        cluster_gene_matrix_opt: DataFrame with K_OPTIMAL cluster means per gene
        gene_linkage: Hierarchical linkage for genes
        atlas_img: Nibabel image object for the brain atlas
        atlas_data: 3D numpy array of atlas parcellation
        genes: List of gene names
        output_path: Base path for output (default: output/figures/supplemental1_optimal)

    Returns:
        fig: Matplotlib figure object
    """
    import pyvista as pv
    import matplotlib.colors as mcolors
    from nilearn import plotting
    from nilearn.image import new_img_like
    from scipy.cluster.hierarchy import dendrogram

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'supplemental1_optimal')

    # Enable off-screen rendering for PyVista
    pv.OFF_SCREEN = True

    n_clusters = config.K_OPTIMAL

    # Get cluster data
    labels_opt = cluster_results[config.K_OPTIMAL]['labels']
    cluster_means_opt = cluster_results[config.K_OPTIMAL]['cluster_means']
    sil_opt = cluster_results[config.K_OPTIMAL]['silhouette']

    # Build parcel mapping from atlas
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0]
    parcel_to_cluster = {}
    for i, parcel in enumerate(sorted(unique_parcels)):
        if i < len(labels_opt):
            parcel_to_cluster[int(parcel)] = labels_opt[i] + 1

    # Find best axial slice for each cluster
    best_slices = {}
    for c in range(1, n_clusters + 1):
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        if cluster_parcels:
            cluster_mask = np.isin(atlas_data, cluster_parcels)
            slice_areas = [np.sum(cluster_mask[:, :, z]) for z in range(atlas_data.shape[2])]
            best_slices[c] = np.argmax(slice_areas)
        else:
            best_slices[c] = atlas_data.shape[2] // 2

    # Create figure: 4 rows x 4 columns
    fig = plt.figure(figsize=(10, 12))

    gs_main = GridSpec(4, 1, figure=fig, height_ratios=[1.2, 0.8, 0.8, 0.8],
                       left=0.06, right=0.94, top=0.95, bottom=0.05, hspace=0.15)

    # Row a: Heatmap
    gs_heatmap = gs_main[0].subgridspec(
        2, 3,
        height_ratios=[0.15, 1],
        width_ratios=[0.05, 1, 0.02],
        hspace=0.02, wspace=0.01
    )

    ax_col_dend = fig.add_subplot(gs_heatmap[0, 1])
    ax_heatmap = fig.add_subplot(gs_heatmap[1, 1])
    ax_colorbar = fig.add_subplot(gs_heatmap[1, 2])

    # Column dendrogram (genes)
    dend_col = dendrogram(
        gene_linkage,
        labels=genes,
        orientation='top',
        leaf_font_size=0,
        color_threshold=0,
        above_threshold_color=config.LINE_COLOR,
        ax=ax_col_dend
    )
    ax_col_dend.axis('off')
    ax_col_dend.text(-0.02, 1.2, 'a', transform=ax_col_dend.transAxes,
                     fontsize=16, fontweight='bold')
    ax_col_dend.set_title(f'K={config.K_OPTIMAL} Gene Expression (Silhouette={sil_opt:.3f}, optimal)',
                          fontsize=11, pad=10)

    gene_order = [genes[i] for i in dend_col['leaves']]
    gene_indices = [genes.index(g) for g in gene_order]

    # Heatmap (no row dendrogram for K_OPTIMAL)
    heatmap_data = cluster_means_opt[:, gene_indices]
    im = ax_heatmap.imshow(
        heatmap_data, aspect='auto', cmap=HEATMAP_CMAP, vmin=-2.5, vmax=2.5
    )

    ax_heatmap.set_yticks(range(n_clusters))
    ax_heatmap.set_yticklabels([f'C{c+1}' for c in range(n_clusters)], fontsize=10)
    ax_heatmap.set_xticks(range(len(gene_order)))
    ax_heatmap.set_xticklabels(gene_order, rotation=45, fontsize=8,
                                ha='right', rotation_mode='anchor')
    ax_heatmap.tick_params(axis='x', length=0)
    plt.colorbar(im, cax=ax_colorbar, label='Z-score')

    # Get affine transform
    affine = atlas_img.affine
    spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
    origin = affine[:3, 3]

    # Row b: 3D brain renderings
    gs_3d = gs_main[1].subgridspec(1, n_clusters, wspace=0.02)
    print('Generating 3D brain renderings for K_OPTIMAL...')
    for c in range(1, n_clusters + 1):
        ax = fig.add_subplot(gs_3d[0, c-1])
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_mask = np.isin(atlas_data, cluster_parcels).astype(float)
        other_mask = np.isin(atlas_data, other_parcels).astype(float)

        plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
        plotter.set_background('white')

        if np.any(other_mask):
            try:
                other_grid = pv.ImageData(dimensions=other_mask.shape, spacing=spacing, origin=origin)
                other_grid.point_data['values'] = other_mask.flatten(order='F')
                other_surface = other_grid.contour([0.5], scalars='values')
                if other_surface.n_points > 0:
                    plotter.add_mesh(other_surface, color='gray', opacity=0.15, smooth_shading=True)
            except Exception:
                pass

        cluster_color = config.CLUSTER_COLORS_OPT[c - 1]
        if np.any(cluster_mask):
            try:
                cluster_grid = pv.ImageData(dimensions=cluster_mask.shape, spacing=spacing, origin=origin)
                cluster_grid.point_data['values'] = cluster_mask.flatten(order='F')
                cluster_surface = cluster_grid.contour([0.5], scalars='values')
                if cluster_surface.n_points > 0:
                    plotter.add_mesh(cluster_surface, color=cluster_color, opacity=0.9, smooth_shading=True)
            except Exception:
                pass

        plotter.camera.position = (563.2, 446.6, 158.8)
        plotter.camera.focal_point = (180.0, -17.0, 15.0)
        plotter.camera.up = (-0.15, -0.18, 0.97)
        plotter.camera.zoom(1.5)

        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax.imshow(img)
        if c == 1:
            ax.set_title(f'C{c}', fontsize=10, fontweight='bold')
            ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold')
        else:
            ax.set_title(f'C{c}', fontsize=10, fontweight='bold')
        ax.axis('off')

    # Row c: Glass brain views
    gs_glass = gs_main[2].subgridspec(1, n_clusters, wspace=0.02)
    print('Generating glass brain views for K_OPTIMAL...')
    for c in range(1, n_clusters + 1):
        ax = fig.add_subplot(gs_glass[0, c-1])
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_vol = np.zeros_like(atlas_data)
        other_vol = np.zeros_like(atlas_data)

        for p in cluster_parcels:
            cluster_vol[atlas_data == p] = 1
        for p in other_parcels:
            other_vol[atlas_data == p] = 0.3

        cluster_nii = new_img_like(atlas_img, cluster_vol)
        other_nii = new_img_like(atlas_img, other_vol)

        cluster_color = config.CLUSTER_COLORS_OPT[c - 1]
        plotting.plot_glass_brain(other_nii, axes=ax, display_mode='z',
                                  cmap='gray', alpha=0.3, colorbar=False,
                                  plot_abs=False, threshold=0.1)
        cluster_cmap = mcolors.LinearSegmentedColormap.from_list(f'c{c}', ['white', cluster_color])
        plotting.plot_glass_brain(cluster_nii, axes=ax, display_mode='z',
                                  cmap=cluster_cmap, alpha=0.9, colorbar=False,
                                  plot_abs=False, threshold=0.5)
        if c == 1:
            ax.text(-0.15, 1.05, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axis('off')

    # Row d: Axial brain slices
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    cluster_grays = {c: 0.3 + 0.4 * (c - 1) / (n_clusters - 1) for c in range(1, n_clusters + 1)}

    gs_axial = gs_main[3].subgridspec(1, n_clusters, wspace=0.02)
    for c in range(1, n_clusters + 1):
        ax = fig.add_subplot(gs_axial[0, c-1])
        z = best_slices[c]
        slice_2d = atlas_data[:, :, z]

        rgb_img = np.ones((*slice_2d.shape, 3))
        cluster_rgb = hex_to_rgb(config.CLUSTER_COLORS_OPT[c - 1])

        for parcel_id in np.unique(slice_2d):
            if parcel_id == 0:
                continue
            mask = slice_2d == parcel_id
            parcel_cluster = parcel_to_cluster.get(int(parcel_id), 0)
            if parcel_cluster == c:
                rgb_img[mask] = cluster_rgb
            elif parcel_cluster > 0:
                gray_val = cluster_grays[parcel_cluster]
                rgb_img[mask] = [gray_val, gray_val, gray_val]

        ax.imshow(np.rot90(rgb_img), aspect='equal')
        ax.set_facecolor('white')
        if c == 1:
            ax.text(-0.15, 1.05, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axis('off')

    # Save
    plt.savefig(output_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path + '.jpg', dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Supplemental Figure 1 (K_OPTIMAL) saved: {output_path}.pdf')

    return fig


# =============================================================================
# FIGURE 2: GENE FAMILY EXPRESSION PROFILES BY CLUSTER
# =============================================================================

def create_figure2_gene_profiles(
    cluster_labels,
    cluster_gene_matrix,
    atlas_img,
    atlas_data,
    output_path=None
):
    """
    Create Figure 2: Gene family expression profiles by cluster.

    This figure shows K_GRANULAR cluster characterization across 4 rows:
      Row 1: 3D brain render (PyVista) - each cluster highlighted
      Row 2: Glass brain projection (nilearn) - axial view
      Row 3: Axial slice at cluster's peak location
      Row 4: Bidirectional radar plot - gene family z-scores

    Args:
        cluster_labels: Array of cluster assignments (0-indexed)
        cluster_gene_matrix: DataFrame with cluster means per gene (rows=clusters, cols=genes)
        atlas_img: Nibabel image object for the brain atlas
        atlas_data: 3D numpy array of atlas parcellation
        output_path: Base path for output (default: output/figures/figure2_gene_profiles)

    Returns:
        fig: Matplotlib figure object
    """
    import pandas as pd
    import pyvista as pv
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from nilearn import plotting
    from nilearn.image import new_img_like

    config.setup_figure_style()
    ensure_output_dirs()

    if output_path is None:
        output_path = os.path.join(config.FIGURES_DIR, 'figure2_gene_profiles')

    # Enable off-screen rendering for PyVista
    pv.OFF_SCREEN = True

    n_clusters = config.K_GRANULAR
    n_families = len(config.CATEGORY_ORDER)

    # Create parcel-to-cluster mapping (1-indexed clusters for display)
    parcel_ids = cluster_gene_matrix.index if hasattr(cluster_gene_matrix.index[0], '__int__') else range(len(cluster_labels))
    # If cluster_gene_matrix rows are C1, C2, etc., we need the original parcel mapping
    # Assume cluster_labels align with expression data indices

    # Build parcel mapping from atlas
    unique_parcels = np.unique(atlas_data)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude background

    # Map parcels to clusters (assumes cluster_labels aligns with parcels 1-332)
    parcel_to_cluster = {}
    for i, parcel in enumerate(sorted(unique_parcels)):
        if i < len(cluster_labels):
            parcel_to_cluster[int(parcel)] = cluster_labels[i] + 1  # 1-indexed

    # Compute family means from cluster_gene_matrix
    family_data = {}
    for family in config.CATEGORY_ORDER:
        genes = config.GENE_FAMILIES.get(family, [])
        genes_present = [g for g in genes if g in cluster_gene_matrix.columns]
        if genes_present:
            family_data[family] = cluster_gene_matrix[genes_present].mean(axis=1)

    family_df = pd.DataFrame(family_data)
    family_df = family_df[config.CATEGORY_ORDER]

    # Find best axial slice for each cluster
    best_slices = {}
    for c in range(1, n_clusters + 1):
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        if cluster_parcels:
            cluster_mask = np.isin(atlas_data, cluster_parcels)
            slice_areas = [np.sum(cluster_mask[:, :, z]) for z in range(atlas_data.shape[2])]
            best_slices[c] = np.argmax(slice_areas)
        else:
            best_slices[c] = atlas_data.shape[2] // 2

    # Color wheel for gene families (9 equally spaced hues)
    family_colors = {}
    for i, fam in enumerate(config.CATEGORY_ORDER):
        hue = i / n_families
        family_colors[fam] = mcolors.hsv_to_rgb([hue, 0.7, 0.9])

    # Create figure: 4 rows x 9 columns
    fig, axes = plt.subplots(4, n_clusters, figsize=(18, 10))
    fig.subplots_adjust(hspace=0.03, wspace=0.02, left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Get affine transform for proper coordinates
    affine = atlas_img.affine
    spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
    origin = affine[:3, 3]

    # Row 1: 3D brain rendering with PyVista
    print('Generating 3D brain renderings...')
    for c in range(1, n_clusters + 1):
        ax = axes[0, c-1]
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_mask = np.isin(atlas_data, cluster_parcels).astype(float)
        other_mask = np.isin(atlas_data, other_parcels).astype(float)

        plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
        plotter.set_background('white')

        # Other clusters as translucent gray
        if np.any(other_mask):
            try:
                other_grid = pv.ImageData(dimensions=other_mask.shape, spacing=spacing, origin=origin)
                other_grid.point_data['values'] = other_mask.flatten(order='F')
                other_surface = other_grid.contour([0.5], scalars='values')
                if other_surface.n_points > 0:
                    plotter.add_mesh(other_surface, color='gray', opacity=0.15, smooth_shading=True)
            except Exception:
                pass

        # Target cluster in color
        cluster_color = config.CLUSTER_COLORS_GRAN[c - 1]
        if np.any(cluster_mask):
            try:
                cluster_grid = pv.ImageData(dimensions=cluster_mask.shape, spacing=spacing, origin=origin)
                cluster_grid.point_data['values'] = cluster_mask.flatten(order='F')
                cluster_surface = cluster_grid.contour([0.5], scalars='values')
                if cluster_surface.n_points > 0:
                    plotter.add_mesh(cluster_surface, color=cluster_color, opacity=0.9, smooth_shading=True)
            except Exception:
                pass

        # Camera position
        plotter.camera.position = (563.2, 446.6, 158.8)
        plotter.camera.focal_point = (180.0, -17.0, 15.0)
        plotter.camera.up = (-0.15, -0.18, 0.97)
        plotter.camera.zoom(1.5)

        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax.imshow(img)
        ax.set_title(f'C{c}', fontsize=10, fontweight='bold')
        ax.axis('off')

    # Row 2: Glass brain views
    print('Generating glass brain views...')
    for c in range(1, n_clusters + 1):
        ax = axes[1, c-1]
        cluster_parcels = [p for p, cl in parcel_to_cluster.items() if cl == c]
        other_parcels = [p for p, cl in parcel_to_cluster.items() if cl != c]

        cluster_vol = np.zeros_like(atlas_data)
        other_vol = np.zeros_like(atlas_data)

        for p in cluster_parcels:
            cluster_vol[atlas_data == p] = 1
        for p in other_parcels:
            other_vol[atlas_data == p] = 0.3

        cluster_nii = new_img_like(atlas_img, cluster_vol)
        other_nii = new_img_like(atlas_img, other_vol)

        cluster_color = config.CLUSTER_COLORS_GRAN[c - 1]
        plotting.plot_glass_brain(other_nii, axes=ax, display_mode='z',
                                  cmap='gray', alpha=0.3, colorbar=False,
                                  plot_abs=False, threshold=0.1)
        cluster_cmap = mcolors.LinearSegmentedColormap.from_list(f'c{c}', ['white', cluster_color])
        plotting.plot_glass_brain(cluster_nii, axes=ax, display_mode='z',
                                  cmap=cluster_cmap, alpha=0.9, colorbar=False,
                                  plot_abs=False, threshold=0.5)
        ax.axis('off')

    # Row 3: Axial brain slices
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    cluster_grays = {c: 0.3 + 0.4 * (c - 1) / (n_clusters - 1) for c in range(1, n_clusters + 1)}

    for c in range(1, n_clusters + 1):
        ax = axes[2, c-1]
        z = best_slices[c]
        slice_2d = atlas_data[:, :, z]

        rgb_img = np.ones((*slice_2d.shape, 3))
        cluster_rgb = hex_to_rgb(config.CLUSTER_COLORS_GRAN[c - 1])

        for parcel_id in np.unique(slice_2d):
            if parcel_id == 0:
                continue
            mask = slice_2d == parcel_id
            parcel_cluster = parcel_to_cluster.get(int(parcel_id), 0)
            if parcel_cluster == c:
                rgb_img[mask] = cluster_rgb
            elif parcel_cluster > 0:
                gray_val = cluster_grays[parcel_cluster]
                rgb_img[mask] = [gray_val, gray_val, gray_val]

        ax.imshow(np.rot90(rgb_img), aspect='equal')
        ax.set_facecolor('white')
        ax.axis('off')

    # Row 4: Radar plots
    # Sigmoid scaling for z-scores
    def apply_scaling(z, zero_r=1.5, scale=1.2):
        return zero_r + np.tanh(z) * scale

    angles = np.linspace(0, 2 * np.pi, n_families, endpoint=False)

    for c in range(1, n_clusters + 1):
        ax = axes[3, c-1]
        ax.set_aspect('equal')
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.axis('off')

        row_name = f'C{c}' if f'C{c}' in family_df.index else family_df.index[c-1]
        values = family_df.loc[row_name].values

        radii = [max(0.1, min(3.0, apply_scaling(v))) for v in values]
        poly_x = [r * np.cos(a) for r, a in zip(radii, angles)]
        poly_y = [r * np.sin(a) for r, a in zip(radii, angles)]

        # z=0 reference circle
        circle = Circle((0, 0), 1.5, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.plot(0, 0, 'ko', markersize=2)

        # Colored triangular segments
        for i in range(n_families):
            i_next = (i + 1) % n_families
            tri_x = [0, poly_x[i], poly_x[i_next], 0]
            tri_y = [0, poly_y[i], poly_y[i_next], 0]
            color1 = np.array(family_colors[config.CATEGORY_ORDER[i]])
            color2 = np.array(family_colors[config.CATEGORY_ORDER[i_next]])
            ax.fill(tri_x, tri_y, color=(color1 + color2) / 2, alpha=0.6, edgecolor='none')

        # Polygon outline
        ax.plot(poly_x + [poly_x[0]], poly_y + [poly_y[0]], 'k-', linewidth=1.5)

        # Data points
        ax.scatter(poly_x, poly_y, c=[family_colors[f] for f in config.CATEGORY_ORDER],
                   s=25, zorder=5, edgecolors='black', linewidth=0.5)

        # Labels
        for i, fam in enumerate(config.CATEGORY_ORDER):
            angle = angles[i]
            x, y = 2.55 * np.cos(angle), 2.55 * np.sin(angle)
            rotation = np.degrees(angle)
            if 90 < rotation < 270:
                rotation += 180
            if c == 1:
                ax.text(x, y, fam, ha='center', va='center', fontsize=12.5,
                        rotation=rotation, color=family_colors[fam], fontweight='bold')
            else:
                ax.text(x, y, fam, ha='center', va='center', fontsize=9.6, rotation=rotation)

    fig.suptitle('Gene Family Expression by Cluster (black circle = z=0; polygon shows bidirectional z-scores)',
                 fontsize=11, fontweight='bold')

    # Save
    plt.savefig(output_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path + '.png', dpi=150, bbox_inches='tight')
    print(f'Figure 2 saved: {output_path}.pdf')

    return fig
