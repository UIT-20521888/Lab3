import numpy as np
import matplotlib.pyplot as plt
import os

def interpolate(history: np.array, label: list) -> np.array:
    return np.interp(label, history[0], history[1]), np.interp(label, history[0], history[2])

def draw_grap(history_DE_32: np.array,
                history_CEM_32: np.array,
                history_DE_1024: np.array, 
                history_CEM_1024: np.array,
                objective_function: str,
                num_parameters: int,
                path_save:str) -> None:
    if num_parameters == 2: 
        label =  [0, 20000, 40000, 60000, 80000, 100000]
    elif num_parameters == 10:
        label =  [0, 200000, 400000, 600000, 800000, 1000000]
    f_DE_32, error_DE_32 = interpolate(history = history_DE_32, label = label)
    f_CEM_32, error_CEM_32 = interpolate(history = history_CEM_32, label = label)
    f_DE_1024, error_DE_1024 = interpolate(history = history_DE_1024, label = label)
    f_CEM_1024, error_CEM_1024 = interpolate(history = history_CEM_1024, label = label)
    # Set graph
    plt.subplots(figsize=(15,10))
    # plot line mean
    plt.plot(label, f_DE_32, color = 'b', label = "DE_128", linewidth = 2)
    plt.plot(label, f_CEM_32, color = 'r', label = "CEM_128", linewidth = 2)
    plt.plot(label, f_DE_1024, color = 'g', label = "DE_1024", linewidth = 2)
    plt.plot(label, f_CEM_1024, color = 'y', label = "CEM_1024", linewidth = 2)
    # plot scatern
    plt.scatter(label, f_DE_32, color = 'b', s = 150)
    plt.scatter(label, f_CEM_32, color = 'r', s = 150)
    plt.scatter(label, f_DE_1024, color = 'g', s = 150)
    plt.scatter(label, f_CEM_1024, color = 'y', s = 150)
    # plot error
    plt.fill_between(label,f_DE_32 - error_DE_32, f_DE_32 + error_DE_32, alpha = 0.2,  color = 'b')
    plt.fill_between(label,f_CEM_32 - error_CEM_32, f_CEM_32 + error_CEM_32, alpha = 0.2,  color = 'r')
    plt.fill_between(label,f_DE_1024 - error_DE_1024, f_DE_1024 + error_DE_1024, alpha = 0.2,  color = 'g')
    plt.fill_between(label,f_CEM_1024 - error_CEM_1024, f_CEM_32 + error_CEM_1024, alpha = 0.2,  color = 'y')
    # set label 
    plt.title(f'Convergence graph with (f = {objective_function},d = {num_parameters})',fontsize = 16, fontweight = 'bold')
    plt.ylabel(f'Objective function',fontsize = 16)
    plt.xlabel(f'Number of Function Evaluations', fontsize = 16)
    plt.legend(loc='upper right')
    file_save = os.path.join(path_save, f'{objective_function}_{num_parameters}_png.png')
    plt.savefig(file_save)
    # plt.show()

if __name__ == '__main__':
    history_DE_32 = np.random.randint(low = 0, high = 10000, size =(3,1000))
    history_CEM_32 = np.random.randint(low = -10, high = 10000, size =(3,1000))
    history_DE_1024 = np.random.randint(low = -10, high = 2000, size =(3,1000))
    history_CEM_1024 = np.random.randint(low = -10, high = 13900, size =(3,1000))
    draw_grap(history_DE_32 = history_DE_32,
                history_CEM_32 = history_CEM_32,
                history_DE_1024 = history_DE_1024,
                history_CEM_1024 = history_CEM_1024,
                objective_function = "Ackly",
                num_parameters = 2,
                path_save = "./")