"""
    Code by: J.COp (Used for Evolve Machine Learner Course)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_svc(X, y, model, support=True):
    """
    Menampilkan Decision Boundary dari SVC
    Apabila tidak ingin menampilkan support, gunakan support=False
    """
    
    # Plot Data
    plt.figure(figsize=(6,6))    
    plt.scatter(X.x_1, X.x_2, c=y, s=5, cmap='bwr')

    # Config awal
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Buat meshgrid
    x = np.linspace(xlim[0], xlim[1], 50)
    y = np.linspace(ylim[0], ylim[1], 50)
    Y, X = np.meshgrid(y, x)
    
    # Hitung nilai dari model
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    val = model.decision_function(xy).reshape(X.shape)
    
    # Gunakan kontur untuk plot Decision Boundary
    if support:
        ax.contour(X, Y, val, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        # Plot support vectornya
        x_sv = model.support_vectors_[:, 0]
        y_sv = model.support_vectors_[:, 1]
        ax.scatter(x_sv, y_sv, s=90, linewidth=1, edgecolor='k', facecolor='None');        
    else:
        ax.contour(X, Y, val, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    
    # Merapikan axis
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def ilustrasi_RBF(X, y, elevation=30, azimuth=30, sigma=1):
    """Ilustrasi untuk contoh RBF"""    
    r = np.exp(-(X ** 2).sum(1)/2/sigma**2)
    plt.figure(figsize=(8,8))    
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X.x_1, X.x_2, r, c=y, s=50, cmap='winter')
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')