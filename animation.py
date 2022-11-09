from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import cv2
import imageio
from utils import sphere, zakharov, rosenbrock, michalewicz, ackley, clip_numbers, DOMAIN
import os

GLOBAL_OPTIMUM = { 'Sphere' : [0, 0, 0],
            'Zakharov' : [0, 0, 0],
            'Rosenbrock' : [1, 1, 0],
            'Michalewicz' : [2.20, 1.57, -1.8013],
            'Ackley' : [0, 0, 0]
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
    
def draw_graph3D(xdata: np.array = [],
                 ydata: np.array = [],
                 zdata: np.array = [],
                 objective_function: str = "Sphere",
                 filename: str = "myfile"):
    minx, maxx = DOMAIN[objective_function]
    x = np.linspace(minx, maxx, 50)
    y = np.linspace(minx, maxx, 50)

    X, Y = np.meshgrid(x, y)

    Z = np.stack((X,Y), axis = 2)
    Z = get_function(objective_function, Z)
    x_global, y_global, z_global = GLOBAL_OPTIMUM[objective_function]
    # plt.show()
    # save file .gif
    
    GEN = 0
    for i in range(1,8):
        if i == 7:
            divide_X, divide_Y, divide_Z = xdata[(i-1)*500:], ydata[(i-1)*500:], zdata[(i-1)*500:]
        else:
            divide_X, divide_Y, divide_Z = xdata[(i-1)*500:i*500], ydata[(i-1)*500:i*500], zdata[(i-1)*500:i*500]
        with imageio.get_writer(f'{filename}_{i}.gif', mode='I', fps = 4) as writer:
            for x, y, z in zip(divide_X, divide_Y, divide_Z):
                # if np.all(z == z_global):
                #     break
                #create grap 3D
                fig = plt.figure(figsize = (10, 10))
                ax = plt.axes(projection='3d')
                # set title
                plt.title(f"{objective_function}\nGEN: {GEN}", fontsize = 30, fontweight = 'bold')
                ax.set_xlabel('x1', fontsize = 15)
                ax.set_ylabel('x2', fontsize = 15)
                ax.set_zlabel('f(x1,x2)',fontsize = 15)
                #set view 
                ax.view_init(42,-179)
                # draw grap
                ax.scatter3D(x_global, y_global, z_global, color = 'b',s = 300, alpha = 1, label = 'Global optimum')
                ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = 'Spectral', edgecolor = 'none', alpha = 0.6)
                
                plt.legend(loc='lower left')
                ax.scatter3D(x, y, z, c = 'black', s = 150, label = 'pop')
                plt.legend(loc = 'upper left')
                #get image from grap
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                    sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                writer.append_data(img)
                ax.cla()
                GEN += 1
            writer.close()

    

if __name__ == '__main__':
    draw_graph3D( xdata = [[1,2,4,5],[1,2]], ydata = [[1,2,3,4],[1,3]], zdata = [[1,2,3,4],[1,5]],objective_function = 'Ackley')