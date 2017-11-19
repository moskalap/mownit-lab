from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import proj3d


def show_plt(sphere, a, name):
    sph = np.dot(sphere, a)
    x = np.array([sph[i, 0] for i in range(len(sph))])
    y = np.array([sph[i, 1] for i in range(len(sph))])
    z = np.array([sph[i, 2] for i in range(len(sph))])
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    u,s,v = linalg.svd(a)
    print(s)
    x_vec = s[0]
    y_vec = s[1]
    z_vec = s[2]

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    a = Arrow3D([0, x_vec], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    a = Arrow3D([0, 0], [0, y_vec], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    a = Arrow3D([0, 0], [0, 0], [0, z_vec], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    ax.set_title(name)
    plt.show()

def zad1_3():
    s = np.linspace(0, 2 * np.pi, 20)
    t = np.linspace(0, np.pi, 20)

    x = np.outer(np.cos(s), np.sin(t)).ravel()
    y = np.outer(np.sin(s), np.sin(t)).ravel()
    z = np.outer(np.ones(np.size(s)), np.cos(t)).ravel()

    matrix = np.array([[x[i], y[i], z[i]] for i in range(len(x))])

    a1_matrix = 100 * np.random.rand(3, 3)
    a2_matrix = 100 * np.random.rand(3, 3)
    a3_matrix = 100 * np.random.rand(3, 3)

    print(a1_matrix)
    show_plt(matrix, a1_matrix, '1')


def show_raw_plt(matrix, name):

    x = np.array([matrix[i, 0] for i in range(len(matrix))])
    y = np.array([matrix[i, 1] for i in range(len(matrix))])
    z = np.array([matrix[i, 2] for i in range(len(matrix))])
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_title(name)
    plt.show()



def zad4():
    import numpy as np
    s = np.linspace(0, 2 * np.pi, 20)
    t = np.linspace(0, np.pi, 20)

    x = np.outer(np.cos(s), np.sin(t)).ravel()
    y = np.outer(np.sin(s), np.sin(t)).ravel()
    z = np.outer(np.ones(np.size(s)), np.cos(t)).ravel()

    matrix = np.array([[x[i], y[i], z[i]] for i in range(len(x))])

    a1_matrix = 100 * np.random.rand(3, 3)
    u,s,v = linalg.svd(a1_matrix)
    while max(s) / min(s) < 100:
        print('asd')
        a1_matrix = 100 * np.random.rand(3, 3)
        u, s, v = linalg.svd(a1_matrix)
    print(s)
    show_plt(matrix, a1_matrix,'100')
    import numpy as np
    show_raw_plt(np.dot(matrix,v), 'SV')
    show_raw_plt(np.dot(matrix, np.dot(np.diag(s), v)), 'SeV')
    show_raw_plt(np.dot(matrix, np.dot(u,np.dot(np.diag(s), v)) ), 'SUeV')

if __name__ == '__main__':
    zad4()






