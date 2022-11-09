from algorithm import DE, CEM 
import numpy as np
def call_mean_std(values: np.array) -> np.array:
    return np.mean(values, axis = 0), np.std(values, axis = 0)
def run_10_times(objective_function: str,
                    num_parameters: int, 
                    mssv: int, 
                    num_individuals: int, 
                    path_file_logger: str, 
                    path_file_git:str) -> np.array:
    random_seeds = [mssv + i for i in range(10)]
    if num_parameters == 2:
        max_of_eval = 100000
    elif num_parameters == 10:
        max_of_eval = 1000000
    history_de, history_cem = [], []

    for random_seed in random_seeds:
        de = DE(num_individuals = num_individuals,
                 num_parameters = num_parameters,
                 objective_function = objective_function,
                 max_of_eval = max_of_eval,
                 random_seed = random_seed)
        cem = CEM(num_individuals = num_individuals,
                    num_parameters = num_parameters,
                    objective_function = objective_function,
                    max_of_eval = max_of_eval,
                    random_seed = random_seed)

        result_de = de.sover(path_file_git, path_file_logger)
        result_cem = cem.sover(path_file_git, path_file_logger)
        history_de.append(result_de)
        history_cem.append(result_cem)
        # if random_seed == 20521889:
        #     break
    # print(history_de, history_cem)
    history_de, history_cem = np.array(history_de), np.array(history_cem)
    # print(history_de.shape, history_cem.shape)
    num_of_eval = history_de[0,:,0]
    means_de, std_de = call_mean_std(history_de[:,:,1])
    means_cem, std_cem = call_mean_std(history_cem[:,:,1])
    # print(means_de.shape, means_cem.shape)
    mean_10_time_de = np.mean(history_de[:,:,1], axis = 0)
    mean_10_time_cem = np.mean(history_cem[:,:,1], axis = 0)
    
    return num_of_eval, means_de, std_de, means_cem, std_cem, mean_10_time_de ,mean_10_time_cem


if __name__ == '__main__':
    run_10_times(objective_function = 'Sphere',
                    num_parameters = 2,
                    mssv = 20521888,
                    num_individuals = 32,
                    path_file_logger = './', 
                    path_file_git = './')
    # print(history_de, history_cem)
    # a = np.array([[[1,2],[3,5],[5,6]],[[4,5],[6,7],[1,2]]])
    # index =  a[0,:,0]
    # value = a[:,:,1]
    # print(a.shape)
    # print(index)
    # print(value)
    # c,d = call_mean_std(a[:,:,1])
    # print(np.mean(value),np.std(value))
    # print(np.mean(c), np.std([[7, 2], [5, 4]]))
