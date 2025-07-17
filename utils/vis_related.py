# %%
import os
import pickle
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.cm as cm

############################################################
############### Plotting for all phenotypes ################
############################################################
def generate_map(map_path, 
                 exist_map: bool = True, 
                 n_components: int = 2, 
                 save_map: bool = False,
                 scale_map: bool = True,
                 map_type: str = "tsne",
                 latent_file_path: Optional[str] = None, 
                 ):
    if exist_map:
        assert os.path.exists(map_path), f"The {map_type} embeddings don't exist"
        # Load embeddings
        map_data = np.load(map_path)
        subj_ids = map_data["subj_id"].reshape(-1)
        map_cls = map_data[f"{map_type}_map_cls"]
        map_avg = map_data[f"{map_type}_map_avg"]
        map_tmp = map_data[f"{map_type}_map_tmp"]
    else:
        # Load and process the latent codes into t-SNE embeddings
        assert latent_file_path is not None, "The path for latent embeddings is not provided"
        latent = np.load(latent_file_path)
        subj_ids = latent["subj_id"].reshape(-1)
        all_tokens = latent["all_token"]
        avg_tokens = np.mean(all_tokens, axis=1)
        cls_tokens = latent["cls_token"]

        if scale_map:
            avg_scaled_data = StandardScaler().fit_transform(avg_tokens)
            cls_scaled_data = StandardScaler().fit_transform(cls_tokens)
            scaled_data = StandardScaler().fit_transform(all_tokens.reshape(-1, all_tokens.shape[2]))
        else:
            avg_scaled_data = avg_tokens
            cls_scaled_data = cls_tokens
            scaled_data = all_tokens.reshape(-1, all_tokens.shape[2])
        
        if map_type == "tsne":
            reducer = TSNE(n_components, perplexity=5, learning_rate="auto")
        elif map_type == "umap":
            reducer = umap.UMAP(n_components)
        map_avg = reducer.fit_transform(avg_scaled_data)
        map_cls = reducer.fit_transform(cls_scaled_data)
        map_tmp = reducer.fit_transform(scaled_data)

        if save_map:
            eval(f"np.savez(map_path, {map_type}_map_avg=map_avg, {map_type}_map_cls=map_cls, {map_type}_map_tmp=map_tmp, subj_id=subj_ids)")
    
    return map_avg, map_cls, map_tmp, subj_ids


def get_phenotype(processed_table_path, subj_ids):
    with open(processed_table_path, 'rb') as file:
        data = pickle.load(file)
    filtered_data = data[data['eid_87802'].isin(subj_ids[:6000])]
    ordered_data = filtered_data.set_index('eid_87802').loc[subj_ids[:6000]].reset_index()
    return ordered_data


def get_label_sep_points(label_name, phenotype_labels):
    probability = [0, 0.2, 0.4, 0.6, 0.8, 1]
    if label_name == "LVCO (L/min)":
        probability = [0, 0.1, 0.3, 0.7, 0.9, 1]
    elif label_name in ['eid_87802', 'Age_group', 'Age_group_idx']:
        return None
    label_sep_points = np.quantile(phenotype_labels, probability, method="closest_observation")
    label_sep_points = [int(p) for p in label_sep_points]
    return label_sep_points


def plot_all_labels(ordered_data, map, map_type: str ="tsne", dim:  int = 2):
    label_names = ordered_data.columns
    for label_name in label_names:
        labels = ordered_data[f'{label_name}'].copy()
        sep_points = get_label_sep_points(label_name, labels)
        if sep_points is None: 
            continue

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
        elif dim == 3:
            # labels_grouped = pd.cut(labels, bins=sep_points, labels=False, include_lowest=True)
            # scatter_data = {
            #     "x": map[:, 0], 
            #     "y": map[:, 1], 
            #     "z": map[:, 2],
            #     f"{label_name}": labels.astype(float),
            #     "group": labels_grouped  # Add the grouped label for color
            # }

            # df_scatter = pd.DataFrame(scatter_data)
            # fig = px.scatter_3d(
            #     df_scatter, 
            #     x='x', y='y', z='z', 
            #     color='group',  # Use the group for coloring
            #     title=f'{map_type} projection for {label_name}',
            #     labels={"group": f'{label_name} Group'},
            #     hover_data={f"{label_name}": True, 'x': False, 'y': False, 'z': False, 'group': False},
            #     color_continuous_scale='Viridis',
            # )
            # fig.update_layout(coloraxis_colorbar=dict(
            #     title=f"{label_name}",
            #     tickvals=[0, 1, 2, 3, 4],  # Labels for each group in the color bar
            #     ticktext=[sep_points[i] for i in range(5)]
            # ))
            # fig.update_traces(mode='markers+text', marker=dict(size=4))
            # fig.show()

        # Assuming `map` and `labels` data are already defined
            sep_points = [0, 1, 2, 3, 4, 5]  # Define separation points
            labels_grouped = pd.cut(labels, bins=sep_points, labels=False, include_lowest=True)

            # Create scatter data as a DataFrame
            scatter_data = {
                "x": map[:, 0], 
                "y": map[:, 1], 
                "z": map[:, 2],
                "label": labels.astype(float),
                "group": labels_grouped  # Add the grouped label for color
            }

            df_scatter = pd.DataFrame(scatter_data)

            # Map colors using Matplotlib colormap (similar to Plotly's 'Viridis')
            colors = cm.viridis(np.linspace(0, 1, len(sep_points) - 1))

            # Create 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot each group with a different color
            for i, color in enumerate(colors):
                group_data = df_scatter[df_scatter["group"] == i]
                ax.scatter(
                    group_data["x"], group_data["y"], group_data["z"],
                    color=color, alpha=0.5, label=f'{sep_points[i]}-{sep_points[i + 1]}', s=4
                )

            # Add color bar legend manually
            cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.5, aspect=5)
            cbar.set_label("Label Groups")
            cbar.set_ticks(np.arange(len(sep_points) - 1))
            cbar.set_ticklabels([f'{sep_points[i]}-{sep_points[i+1]}' for i in range(len(sep_points) - 1)])

            # Set title and show plot
            ax.set_title(f'{map_type} projection for {label_name}')
            ax.legend()
            plt.show()


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

def plot_temporal_tokens(num_tmp, num_subj, map, map_type: str = "tsne", dim: int = 2):
    tmps = np.arange(num_tmp).reshape(1, -1)
    tmp_labels = np.tile(tmps, (num_subj, 1))
    tmp_labels_ = tmp_labels.reshape(-1)

    if dim == 2:
        sep_points = [i for i in np.arange(num_tmp)]
        cmap = plt.get_cmap('Spectral')

        plt.figure(figsize=(10, 6))
        plt.scatter(
            map[:, 0],
            map[:, 1],
            c=tmp_labels_.astype(float), 
            cmap=cmap,
            s=15,
            alpha=1.0)
        cbar = plt.colorbar(boundaries=[i-0.5 for i in np.arange(num_tmp+1)], ticks=sep_points)
        cbar.set_label(f"Time section")

        plt.gca().set_aspect('equal', 'datalim')
        
    elif dim == 3:
        scatter_data = {"x": map[:, 0], 
                        "y": map[:, 1], 
                        "z": map[:, 2], 
                        "time": tmp_labels_.astype(float)}
        df_scatter = pd.DataFrame(scatter_data)
        fig = px.scatter_3d(
            df_scatter, 
            x='x', y='y', z='z', 
            color='time',
            title=f"{map_type} maps for temporal sections",
            labels={'label': 'Time'},
            hover_data={'time': True, 'x': False, 'y': False, 'z': False},
        )
        fig.update_traces(mode='markers+text')
        fig.update_traces(marker=dict(size=1))
        fig.show()
    
    
def plot_subject_tokens_with_phenotype(num_subj, num_tmp, map, 
                                       label_name: str = "LVM (g)",
                                       map_type: str = "tsne", 
                                       dim: int = 2, 
                                       lines: bool = True,
                                       interpolation: bool = True):
    # Create all needed labels such as subject labels
    subj_l = np.arange(num_subj).reshape(-1, 1)
    subj_labels = np.tile(subj_l, (1, num_tmp))
    subj_labels_ = subj_labels.reshape(-1)

    if dim == 2:
        subj_sep_points = list(np.arange(num_subj))
        cmap = plt.get_cmap('Spectral')
        norm = mcolors.BoundaryNorm(subj_sep_points, cmap.N)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            map[:num_subj*num_tmp, 0],
            map[:num_subj*num_tmp, 1],
            c=subj_labels_.astype(float), 
            cmap=cmap,
            norm=norm,
            s=15,
            alpha=1.0)
        cbar = plt.colorbar(boundaries=subj_sep_points, ticks=subj_sep_points)
        cbar.set_label(f"Subject IDs")

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'{map_type} projection for latents of subjects', fontsize=24)

    elif dim == 3:
        # Get phenotype labels and phenotype group labels
        phenotype_all_labels = ordered_data[f"{label_name}"].copy()
        phenotype_labels = np.array(phenotype_all_labels[:num_subj]).reshape(-1, 1)
        phenotype_labels = np.tile(phenotype_labels, (1, num_tmp))
        phenotype_labels_ = phenotype_labels.reshape(-1)

        label_sep_points = get_label_sep_points(label_name, phenotype_all_labels)
        if label_sep_points is None: 
            return
        phenotype_labels_groups_ = pd.cut(phenotype_labels_, bins=label_sep_points, labels=False, include_lowest=True)
        
        # Generate scatter dataframe
        scatter_data = {"x": map[:num_subj*num_tmp, 0], 
                        "y": map[:num_subj*num_tmp, 1], 
                        "z": map[:num_subj*num_tmp, 2], 
                        "subj": subj_labels_.astype(float),
                        f"{label_name}": phenotype_labels_,
                        f"{label_name}_group": phenotype_labels_groups_,
                        }
        df_scatter = pd.DataFrame(scatter_data)
        
        # Initialize a blank figure
        fig = go.Figure()
        c = px.colors.qualitative.Plotly  # Get distinct colors
        group_c = px.colors.qualitative.Vivid  # Get distinct colors
        subject_colors = {i: c[i % len(c)] for i in range(num_subj)} 
        group_colors = {i: group_c[i % len(group_c)] for i in range(len(label_sep_points) - 1)} 
        
        # Add a dummy trace for each group to show in the legend
        for group_id, color in group_colors.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers+lines",
                    marker=dict(size=5, color=color),
                    name=f"{label_name} group {group_id}",
                    showlegend=True
                )
            )
        # Loop through each subject to create a separate trace for their points and lines
        for i in range(num_subj):
            subject_data = df_scatter.iloc[i * num_tmp:(i + 1) * num_tmp]  # Get data for each subject
            phenotype_labels = subject_data[f"{label_name}"].values
            phenotype_label_groups = subject_data[f"{label_name}_group"].values
            assert len(set(phenotype_labels)) == 1
            assert len(set(phenotype_label_groups)) == 1
            phenotype_label = phenotype_labels[0]
            phenotype_label_group = phenotype_label_groups[0]
            
            subject_color = subject_colors[i] 
            group_color = group_colors[phenotype_label_group] 
            
            # Extract subject coordinates
            x_points = subject_data["x"].values
            y_points = subject_data["y"].values
            z_points = subject_data["z"].values

            # Close the loop by adding the first point to the end of each coordinate array
            x_points = np.append(x_points, x_points[0])
            y_points = np.append(y_points, y_points[0])
            z_points = np.append(z_points, z_points[0])

            if interpolation:
                # Create an array of parameter values for interpolation
                t = np.arange(len(x_points))
                t_smooth = np.linspace(t.min(), t.max(), 100)  # Increase number of points for smoothness

                # Interpolate x, y, and z coordinates
                x_points = interp1d(t, x_points, kind='cubic')(t_smooth)
                y_points = interp1d(t, y_points, kind='cubic')(t_smooth)
                z_points = interp1d(t, z_points, kind='cubic')(t_smooth)
            
            hover_text = [f"Subject {i+1}<br>{label_name} {phenotype_label}<br>group {phenotype_label_group}" 
                        for _ in range(len(x_points)-1)]
            mode = 'lines+markers' if lines else 'markers'
            fig.add_trace(
                go.Scatter3d(
                    x=x_points,
                    y=y_points,
                    z=z_points,
                    mode=mode,
                    marker=dict(
                        size=2,
                        color=group_color,  # Color by subject label
                        opacity=0.5
                    ),
                    line=dict(
                        color=group_color,  # Use the color of the first point in the subject group
                        colorscale="Viridis",
                        width=2,
                    ),
                    text=hover_text,  # Add hover text for each point
                    hoverinfo="text",
                    showlegend=False
                    )
                )

        # Set layout options
        fig.update_layout(
            title=f"{map_type} maps for latents of subjects with phenotype {label_name}",
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis"
            )
        )

        fig.show()
        

def plot_perimeter_with_phenotype(ordered_data, map, map_type, num_subj, num_tmp, 
                                  normalize: bool = True,
                                  phenotype_list: Optional[list] = None,
                                  ):
    def calculate_perimeter(points):
        distances = np.linalg.norm(np.diff(points, axis=1, prepend=points[:, -1:]), axis=2)
        perimeters = distances.sum(axis=1)
        return perimeters
    
    if phenotype_list is None:
        phenotype_list = ordered_data.columns
        
    for label_name in phenotype_list:
        phenotype_all_labels = ordered_data[f"{label_name}"].copy()
        phenotype_labels = np.array(phenotype_all_labels[:num_subj]).reshape(-1)

        label_sep_points = get_label_sep_points(label_name, phenotype_labels)
        if label_sep_points is None:
            continue

        points = map[:num_subj * num_tmp].reshape(-1, num_tmp, map.shape[-1])
        perimeters = calculate_perimeter(points)

        # Keep the original values for later use
        original_X = perimeters.reshape(-1, 1)
        original_y = phenotype_labels.reshape(-1, 1)

        # Normalize if needed
        if normalize:
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X = scaler_X.fit_transform(original_X)
            y = scaler_y.fit_transform(original_y)
        else:
            X = original_X
            y = original_y
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)

        # Create a DataFrame for seaborn
        import pandas as pd
        data = pd.DataFrame({
            'Perimeters': X.flatten(),
            label_name: y.flatten()
        })

        # Plot with sns.jointplot
        g = sns.jointplot(
            data=data, x='Perimeters', y=label_name, kind='scatter', height=8, ratio=5, marginal_kws={'bins': 30, 'kde': True}
        )

        # Add the regression line
        g.ax_joint.plot(X_range, y_pred, color='red', linewidth=2, label='Learned Regression Line')

        # Replace the axis ticks with unnormalized values
        xticks = g.ax_joint.get_xticks()
        yticks = g.ax_joint.get_yticks()
        g.ax_joint.set_xticklabels([f'{scaler_X.inverse_transform([[tick]])[0][0]:.2f}' for tick in xticks], rotation=45)
        g.ax_joint.set_yticklabels([f'{scaler_y.inverse_transform([[tick]])[0][0]:.2f}' for tick in yticks])

        # Set the font size for axis titles
        g.ax_joint.set_xlabel('Perimeters', fontsize=20)
        g.ax_joint.set_ylabel(label_name, fontsize=20)
        g.ax_marg_x.set_xlabel('Perimeters', fontsize=20)
        g.ax_marg_y.set_ylabel(label_name, fontsize=20)
        
        g.ax_joint.legend(fontsize=14)
        # g.fig.suptitle(f'{map_type} regression for {label_name}', y=1.03, fontsize=24)
        plt.show()


def plot_scatter(map, labels, label_name, dim: int = 3, map_type: str = "tsne"):
    if dim == 2:
        unique_labels = np.unique(health_binary_labels)  # Find unique labels
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        label_color_map = dict(zip(unique_labels, colors))

        plt.figure(figsize=(10, 6))
        for label in unique_labels:
            mask = (health_binary_labels == label)
            plt.scatter(map[mask, 0], map[mask, 1], label=label, color=label_color_map[label], s=15, alpha=0.5)

        plt.legend(title='Labels')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'{map_type} projection for {label_name}', fontsize=24)
        plt.show()
    elif dim == 3:
        scatter_data = {
            "x": map[:, 0], 
            "y": map[:, 1], 
            "z": map[:, 2],
            f"{label_name}": labels  # Add the grouped label for color
        }

        df_scatter = pd.DataFrame(scatter_data)
        fig = px.scatter_3d(
            df_scatter, 
            x='x', y='y', z='z', 
            color=f"{label_name}",  # Use the group for coloring
            title=f'{map_type} projection for {label_name}',
            labels={f"{label_name}": f'{label_name} Group'},
            hover_data={f"{label_name}": True, 'x': False, 'y': False, 'z': False},
            )
        fig.update_layout(
            legend_title=f'{label_name} Group',
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        fig.update_traces(marker=dict(size=2))
        fig.show()
        

def rotate_points(points, angle_deg, axis='z'):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'z':  # Rotation around the z-axis
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'x':  # Rotation around the x-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':  # Rotation around the y-axis
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    
    return np.dot(points, rotation_matrix.T)


def calculate_centroid(group_data):
# Function to calculate the centroid of each group
    return group_data[['x', 'y', 'z']].mean(axis=0).values
