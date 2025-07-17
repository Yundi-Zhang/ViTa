import numpy as np
from matplotlib import pyplot as plt
import torch
import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px


def get_label_sep_points(label_name, phenotype_labels):
    probability = [0, 0.2, 0.4, 0.6, 0.8, 1]
    if label_name == "LVCO (L/min)":
        probability = [0, 0.1, 0.3, 0.7, 0.9, 1]
    elif label_name in ['eid_87802', 'Age_group', 'Age_group_idx']:
        return None
    label_sep_points = np.quantile(phenotype_labels, probability, method="closest_observation")
    label_sep_points = [int(p) for p in label_sep_points]
    return label_sep_points


def plot_one_label(map, map_type: str ="tsne", 
                   ordered_data = None, labels = None,
                   label_name: str = 'LVM (g)', dim: int = 2, ):
    if labels is None:
        labels = ordered_data[f"{label_name}"].copy()
    if isinstance(labels, torch.Tensor):
        labels = np.array(labels)
    sep_points = get_label_sep_points(label_name, labels)
    if sep_points is None: 
        print("This phenotype is skiped for visualization.")
        return

    if dim == 2:
        cmap = plt.get_cmap('Spectral')
        norm = mcolors.BoundaryNorm(sep_points, cmap.N)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            map[:, 0],
            map[:, 1],
            c=labels.astype(float), 
            cmap=cmap,
            norm=norm,
            s=15,
            alpha=0.5)
        cbar = plt.colorbar(boundaries=sep_points, ticks=sep_points)
        cbar.set_label(f"{label_name}")

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'{map_type} projection for {label_name}', fontsize=24)
        return plt
    elif dim == 3:
        labels_grouped = pd.cut(labels, bins=sep_points, labels=False, include_lowest=True)
        scatter_data = {
            "x": map[:, 0], 
            "y": map[:, 1], 
            "z": map[:, 2],
            f"{label_name}": labels.astype(float),
            "group": labels_grouped  # Add the grouped label for color
        }

        df_scatter = pd.DataFrame(scatter_data)
        fig = px.scatter_3d(
            df_scatter, 
            x='x', y='y', z='z', 
            color='group',  # Use the group for coloring
            title=f'{map_type} projection for {label_name}',
            labels={"group": f'{label_name} Group'},
            hover_data={f"{label_name}": True, 'x': False, 'y': False, 'z': False, 'group': False},
            color_continuous_scale='Viridis',
        )
        fig.update_layout(coloraxis_colorbar=dict(
            title=f"{label_name}",
            tickvals=[0, 1, 2, 3, 4],  # Labels for each group in the color bar
            ticktext=[sep_points[i] for i in range(5)]
        ))
        fig.update_traces(mode='markers+text', marker=dict(size=2))
        fig.show()
        return fig
