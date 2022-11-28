from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import cv2
import imageio
from utils import sphere, zakharov, rosenbrock, michalewicz, ackley, clip_numbers, DOMAIN
import os
import json
import sys
import gc

GLOBAL_OPTIMUM = {'Sphere': [0, 0, 0],
                  'Zakharov': [0, 0, 0],
                  'Rosenbrock': [1, 1, 0],
                  'Michalewicz': [2.20, 1.57, -1.8013],
                  'Ackley': [0, 0, 0]
                  }


def get_function(objective_function: str = 'Sphere', Z: np.array = []) -> np.array:
    if objective_function == 'Sphere':
        _objective_function = sphere
    elif objective_function == 'Zakharov':
        _objective_function = zakharov
    elif objective_function == 'Rosenbrock':
        _objective_function = rosenbrock
    elif objective_function == 'Michalewicz':
        _objective_function = michalewicz
    elif objective_function == 'Ackley':
        _objective_function = ackley
    else:
        print("Please check the objective function, only in ('Sphere', 'Zakharov', 'Rosenbrock','Michalewicz', 'Ackley')")
        return
    results = []
    for i in Z:
        data = []
        for j in i:
            scores = _objective_function(j)
            data.append(scores)
        results.append(data)
    return np.array(results)


def draw(filename: str, i: int,
         divide_X: np.array,
         divide_Y: np.array,
         divide_Z: np.array,
         objective_function: str,
         GEN: int,
         x_global: float,
         y_global: float,
         z_global: float,
         X, Y, Z) -> int:
    with imageio.get_writer(f'{filename}_{i}.gif', mode='I', fps=4) as writer:
        for x, y, z in zip(divide_X, divide_Y, divide_Z):
            # if np.all(z == z_global):
            #     break
            # create grap 3D
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            # set title
            plt.title(f"{objective_function}\nGEN: {GEN}",
                      fontsize=30, fontweight='bold')
            ax.set_xlabel('x1', fontsize=15)
            ax.set_ylabel('x2', fontsize=15)
            ax.set_zlabel('f(x1,x2)', fontsize=15)
            # set view
            ax.view_init(42, -179)
            # draw grap
            ax.scatter3D(x_global, y_global, z_global, color='b',
                         s=300, alpha=1, label='Global optimum')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='Spectral', edgecolor='none', alpha=0.6)

            plt.legend(loc='lower left')
            ax.scatter3D(x, y, z, c='black', s=150, label='pop')
            plt.legend(loc='upper left')
            # get image from grap
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            writer.append_data(img)
            ax.cla()
            GEN += 1
            # del img, fig, ax
            # gc.collect()

    writer.close()
    # del writer
    # gc.collect()
    return GEN


def draw_graph3D(xdata: np.array = [],
                 ydata: np.array = [],
                 zdata: np.array = [],
                 objective_function: str = "Sphere",
                 filename: str = "myfile",
                 start: int = 0,
                 end: int = 0):
    minx, maxx = DOMAIN[objective_function]
    x = np.linspace(minx, maxx, 50)
    y = np.linspace(minx, maxx, 50)

    X, Y = np.meshgrid(x, y)

    Z = np.stack((X, Y), axis=2)
    Z = get_function(objective_function, Z)
    x_global, y_global, z_global = GLOBAL_OPTIMUM[objective_function]
    # plt.show()
    # save file .gif

    GEN = (start-1)*500
    for i in range(start, end):
        print(GEN)
        if i == 7:
            divide_X, divide_Y, divide_Z = xdata[(
                i-1)*500:], ydata[(i-1)*500:], zdata[(i-1)*500:]
        else:
            divide_X, divide_Y, divide_Z = xdata[(
                i-1)*500:i*500], ydata[(i-1)*500:i*500], zdata[(i-1)*500:i*500]
        GEN = draw(filename=filename, i=i,
                   divide_X=divide_X,
                   divide_Y=divide_Y,
                   divide_Z=divide_Z,
                   objective_function=objective_function,
                   GEN=GEN,
                   x_global=x_global,
                   y_global=y_global,
                   z_global=z_global,
                   X=X,
                   Y=Y,
                   Z=Z)


def read_file_gif(path_file: str) -> np.array:
    with open(path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    data = np.array(data['history'])
    x_data, y_data, z_data = [], [], []
    for gen, value in enumerate(data):
        x, y, z = [], [], []
        for pid in value[f'#GEN_{gen}']:
            x.append(pid['x1'])
            y.append(pid['y1'])
            z.append(pid['z1'])
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)

    return np.array(x_data), np.array(y_data), np.array(z_data)


if __name__ == '__main__':
    path_save = sys.argv[4]
    path_gif = sys.argv[2]
    start = int(sys.argv[6])
    end = int(sys.argv[8])
    # print(path_save,path_gif,start,end)
    x_data, y_data, z_data = read_file_gif(path_gif)
    file = path_gif.split('/')[-1]
    path_foder = os.path.join(path_save, file.split('.')[0])
    if os.path.exists(path_foder) == False:
        os.mkdir(path_foder)    
    path_file_save = os.path.join(path_foder, file.split('.')[0])
    objective_function = file.split('.')[0].split('_')[1]
    print(f"[#INFO]{objective_function}")
    draw_graph3D(x_data, y_data, z_data, objective_function, path_file_save, start,end)
    # print(path_save, path_gif)