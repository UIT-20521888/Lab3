from bisection import run_10_times
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from draw_graph import draw_grap
import shutil

MSSV = 20521888
LIST_POP = [32, 64, 128, 256, 512, 1024]
LIST_D = [2, 10]
LIST_FUNCTIONS =  ['Sphere', "Zakharov" , 'Rosenbrock', 'Michalewicz', 'Ackley']

def run_for_object_function(object_function: list  = LIST_FUNCTIONS):
    path_gif = './File_gif'
    path_images = './File_images'
    path_log = './File_log'
    path_result = './File_result'
    if os.path.exists(path_gif) == True:
        shutil.rmtree(path_gif)
    if os.path.exists(path_images) == True:
        shutil.rmtree(path_images)
    if os.path.exists(path_log) == True:
        shutil.rmtree(path_log)
    if os.path.exists(path_result) == True:
        shutil.rmtree(path_result)
    os.mkdir(path_gif)
    os.mkdir(path_images)
    os.mkdir(path_log)
    os.mkdir(path_result)

    for function in object_function:
        run_for_object_d(path_log = path_log, 
                            path_images = path_images, 
                            path_gif = path_gif, 
                            path_result = path_result, 
                            object_function = function)

def run_for_object_d(path_log: str, path_images: str, path_gif:str, path_result:str, object_function, list_d:list = LIST_D):
    for d in list_d:
        run_for_N(path_log = path_log, 
                    path_gif = path_gif, 
                    object_function = object_function, 
                    num_parameters = d, 
                    path_images = path_images, 
                    path_result = path_result)

def run_for_N(path_log: str, path_gif: str,
             object_function: str, num_parameters: int,
             path_images : str, path_result :str,
             list_pop: list = LIST_POP):

    history_DE_32, history_DE_1024, history_CEM_32, history_CEM_1024 = None, None, None, None
    list_de, list_cem = [], []

    for pop in list_pop:
        num_of_eval, means_de, std_de, means_cem, std_cem ,mean_10_time_de ,mean_10_time_cem = run_10_times(objective_function = object_function,
                                                                                                            num_parameters = num_parameters, 
                                                                                                            mssv = MSSV, 
                                                                                                            num_individuals = pop,
                                                                                                            path_file_git = path_gif,
                                                                                                            path_file_logger = path_log)
        
        if pop == 32:
            history_DE_32, history_CEM_32 = [num_of_eval,means_de, std_de], [num_of_eval, means_cem, std_cem]
        if pop == 1024:
            history_DE_1024, history_CEM_1024 = [num_of_eval, means_de, std_de], [num_of_eval, means_cem, std_cem]
        _, p = ttest_ind(mean_10_time_de ,mean_10_time_cem)
        if p <= 0.05:
            if np.mean(mean_10_time_de) < np.mean(mean_10_time_cem):
                list_de.append(f"{np.mean(mean_10_time_de)} ({np.std(mean_10_time_de)})*")
                list_cem.append(f"{np.mean(mean_10_time_cem)} ({np.std(mean_10_time_cem)})")
            else:
                list_de.append(f"{np.mean(mean_10_time_de)} ({np.std(mean_10_time_de)})")
                list_cem.append(f"{np.mean(mean_10_time_cem)} ({np.std(mean_10_time_cem)})*")
        else:
            list_de.append(f"{np.mean(mean_10_time_de)} ({np.std(mean_10_time_de)})")
            list_cem.append(f"{np.mean(mean_10_time_cem)} ({np.std(mean_10_time_cem)})")

    history_DE_32, history_CEM_32 = np.array(history_DE_32), np.array(history_CEM_32)
    history_DE_1024, history_CEM_1024 = np.array(history_DE_1024), np.array(history_CEM_1024)
    #draw graph with (f,d)
    draw_grap(history_DE_32 = history_DE_32,
                history_CEM_32 = history_CEM_32,
                history_DE_1024 = history_DE_1024, 
                history_CEM_1024 = history_CEM_1024,
                objective_function = object_function,
                num_parameters = num_parameters,
                path_save = path_images)
    # save info
    list_pop_2 = list_pop.copy()
    list_pop_2.append("* is where it is statistically significant and should be bolded")
    list_de.append("")
    list_cem.append("")
    path_save = os.path.join(path_result,f"{object_function}_{num_parameters}.xlsx")
    dic = {"Popsize":list_pop_2,
            "DE": list_de,
            "CEM": list_cem}
    data_df = pd.DataFrame(dic)
    # print(dic)
    data_df.to_excel(path_save, index = False)

if __name__ == '__main__':
    # run_for_object_function()
    # run_for_N(path_log = "./", path_gif ="./",
    #          object_function = "Ackley", num_parameters = 2 ,
    #          path_images = "./", path_result = "./", list_pop =[32])
    # _, p = ttest_ind([1,2,3,4],[3,4,5,6])
    # print(p)
    run_for_object_function()