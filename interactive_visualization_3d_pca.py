import numpy as np
import plotly.graph_objects as go
import os
import webbrowser
import math
from starter_pack.src.data_loader import DataLoader

# Load data
loader = DataLoader()
digits = loader.load_digits()
X = digits['X']
y = digits['y']

print("Computing PCA...")
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_pca_3d = U[:, :3] * S[:3]

# Calculate explained variance
eigenvalues = S**2 / (len(S) - 1)
explained_var = eigenvalues[:3] / np.sum(eigenvalues)
print(f"Explained variance: PC1={explained_var[0]:.3f} ({explained_var[0]*100:.1f}%), "
      f"PC2={explained_var[1]:.3f} ({explained_var[1]*100:.1f}%), "
      f"PC3={explained_var[2]:.3f} ({explained_var[2]*100:.1f}%)")

print("Creating interactive 3D visualization of PCA dataset")

# TAB10 COLORS (matplotlib's default colormap)
tab10_colors = [
    '#1f77b4',  # 0 - Blue
    '#ff7f0e',  # 1 - Orange
    '#2ca02c',  # 2 - Green
    '#d62728',  # 3 - Red
    '#9467bd',  # 4 - Purple
    '#8c564b',  # 5 - Brown
    '#e377c2',  # 6 - Pink
    '#7f7f7f',  # 7 - Gray
    '#bcbd22',  # 8 - Yellow-green
    '#17becf'   # 9 - Cyan
]

# Create figure
fig = go.Figure()

for digit in range(10):
    mask = y == digit
    fig.add_trace(go.Scatter3d(
        x=X_pca_3d[mask, 0],
        y=X_pca_3d[mask, 1],
        z=X_pca_3d[mask, 2],
        mode='markers',
        name=f'Digit {digit}',
        marker=dict(
            size=5,
            color=tab10_colors[digit],
            opacity=0.7,
            line=dict(width=0.5, color='black')
        ),
        text=[f'Digit: {digit}' for _ in range(np.sum(mask))],
        hoverinfo='text'
    ))

# Set view angle to match matplotlib's elev=20, azim=45
elev_rad = 20 * math.pi / 180
azim_rad = 45 * math.pi / 180
distance = 2.0

eye_x = distance * math.cos(azim_rad) * math.cos(elev_rad)
eye_y = distance * math.sin(azim_rad) * math.cos(elev_rad)
eye_z = distance * math.sin(elev_rad)

fig.update_layout(
    title=dict(
        text=f'PCA 3D Visualization of Digits<br>'
             f'<sup>Explained variance: PC1={explained_var[0]:.1%}, '
             f'PC2={explained_var[1]:.1%}, PC3={explained_var[2]:.1%}</sup>',
        font=dict(size=14),
        x=0.5
    ),
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
        camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='data'
    ),
    legend=dict(
        title='Digit Label',
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='left',
        x=1.02,
        bgcolor='rgba(255, 255, 255, 0.85)',
        bordercolor='black',
        borderwidth=1,
        font=dict(size=10)
    ),
    width=1000,
    height=800,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Save to HTML - CREATE DIRECTORY IF NEEDED
output_path = 'additional/track_a_pca3d_interactive.html'

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Check if file exists
if os.path.exists(output_path):
    print(f"Warning: {output_path} already exists and will be overwritten.")

fig.write_html(output_path)
print(f"Saved to: {output_path}")

# Open in browser
webbrowser.open(f'file://{os.path.abspath(output_path)}')
print("Opened in browser! Rotate, zoom, and hover over points.\nThe labels show the digit class. Interactive-ish")