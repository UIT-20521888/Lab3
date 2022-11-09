from utils import sphere, zakharov, rosenbrock, michalewicz, ackley, clip_numbers, DOMAIN
import numpy as np
from animation import draw_graph3D
import os

class DE:
    def __init__(self, num_individuals: int = 32,
                 num_parameters: int = 2,
                  objective_function: str = 'Sphere',
                  max_of_eval: int = 100000,
                  random_seed: int = 20521888,
                  scale: float = 0.8,
                  cross: float = 0.9) -> None:

        self.num_individuals = num_individuals
        self.num_parameters = num_parameters
        self.random_seed = random_seed
        self.max_of_eval = max_of_eval
        self.num_of_eval = 0
        self.best_finess = 0
        self.scale = scale
        self.cross = cross
        self.best_answer = None
        self.objective_function = objective_function

        if objective_function == "Sphere":
            self._objectsfunction = sphere
        elif objective_function == "Zakharov": 
            self._objectsfunction = zakharov
        elif objective_function == "Rosenbrock":
            self._objectsfunction = rosenbrock
        elif objective_function == "Michalewicz":
            self._objectsfunction = michalewicz
        elif objective_function == "Ackley":
            self._objectsfunction = ackley
        else:
            print("Please check the objective function, only in ('Sphere', 'Zakharov', 'Rosenbrock', 'Ackley')")
            exit(1)

    def _init_pop(self) -> np.array:
        x_min, x_max = DOMAIN[self.objective_function]
        pop = np.random.rand(self.num_individuals, self.num_parameters) * (x_max - x_min) + x_min
        return pop

    def mutant_population(self,fitness: np.array ,pop: np.array) -> np.array:
        
        ind_offset = []
        fitness_offset = []

        for index in range(self.num_individuals):
            candidates = list(range(0, self.num_individuals))
            np.random.shuffle(candidates)
            candidates.remove(index)
            r0, r1,r2 = np.random.choice(candidates, 3, replace = False)
            x0, x1, x2 = pop[r0], pop[r1], pop[r2]
            x_diff = clip_numbers(x0 + self.scale * (x1 - x2), self.objective_function)
            # vector_mutan.append(x_diff)
            cross_poin = np.random.rand(self.num_parameters) < self.cross
            if not cross_poin.any():
                cross_poin[np.random.randint(0, self.num_parameters)] = True
            vector_cross = np.where(cross_poin, x_diff, pop[index])
            # print(cross_poin, x_diff, pop[index], vector_cross)
            vector_cross = clip_numbers(vector_cross, self.objective_function)

            score = self._objectsfunction(vector_cross)

            ind_offset.append(pop[index])
            fitness_offset.append(fitness[index])
            if score < fitness[index]:
                ind_offset[-1] = vector_cross
                fitness_offset[-1] = score

        return np.array(ind_offset), np.array(fitness_offset)
    def sover(self, path_file_gif, path_file_logger):
        np.random.seed(self.random_seed)

        pop = self._init_pop()
        fitness = [self._objectsfunction(x) for x in pop]
        GEN = 0
        num_of_eval = self.num_individuals

        best_ind = np.argmin(fitness)
        best = fitness[best_ind]

        history_pop, history_fitness = [pop], [fitness]
        history_best_eval = [[num_of_eval, best]]
        
        print(f"#GEN: {GEN}\n\tBest-ind: { pop[best_ind] }\n\tFitness: {best}")
        file_logger = os.path.join(path_file_logger, f"DE_{self.objective_function}_{self.num_parameters}_{self.num_individuals}_{self.random_seed}.txt")
        f = open(file_logger, 'w+',encoding = 'utf-8')
        f.write("#GEN\tx_best\tf(x_best)\t#eval\n")
        f.write(f"{ GEN }\t{ pop[best_ind] }\t{ best }\t{ num_of_eval }\n")

        while True:
            pop, fitness = self.mutant_population(fitness,pop)
            best_ind = np.argmin(fitness)
            best = fitness[best_ind]
            GEN += 1
            history_pop.append(pop)
            history_fitness.append(fitness)
            num_of_eval += self.num_individuals
            history_best_eval.append([num_of_eval, best])

            print(f"#GEN: { GEN }\n\tBest-ind: { pop[best_ind] }\n\tFitness: { best }")
            f.write(f"{ GEN }\t{ pop[best_ind] }\t{ best }\t{ num_of_eval }\n")

            if num_of_eval >= self.max_of_eval:
                break
        
        history_pop, history_fitness = np.array(history_pop), np.array(history_fitness)
        f.write(f"#Result: mean = {np.mean(np.array(history_best_eval)[:,1])}\tstd = {np.std(np.array(history_best_eval)[:,1])}")
        f.close()
        # print(history_pop.shape)
        # if self.num_parameters == 2 and self.num_individuals == 32 and self.random_seed == 20521888:
        #     file_name = os.path.join(path_file_gif,f"DE_{self.objective_function}_{self.num_parameters}")
        #     draw_graph3D(xdata = history_pop[:,:,0] ,
        #          ydata = history_pop[:,:,1],
        #          zdata = history_fitness,
        #          objective_function = self.objective_function,
        #          filename = file_name)
        
        return history_best_eval

class CEM:
    def __init__(self, num_individuals: int = 32,
                        num_parameters: int = 2,
                        objective_function: str = 'Sphere',
                        max_of_eval: int = 100000,
                        random_seed: int = 20521888,
                        init_singma: int = 5,
                        init_epsilon: int = 0.1,
                        num_selection: int = 10):
        self.num_individuals = num_individuals
        self.num_parameters = num_parameters
        self.num_selection = num_selection
        self.objective_function = objective_function
        self.max_of_eval = max_of_eval
        self.random_seed = random_seed
        self.init_singma = init_singma
        self.initi_epsilon = init_epsilon

        if objective_function == "Sphere":
            self._objectsfunction = sphere
        elif objective_function == "Zakharov": 
            self._objectsfunction = zakharov
        elif objective_function == "Rosenbrock":
            self._objectsfunction = rosenbrock
        elif objective_function == "Michalewicz":
            self._objectsfunction = michalewicz
        elif objective_function == "Ackley":
            self._objectsfunction = ackley
        else:
            print("Please check the objective function, only in ('Sphere', 'Zakharov', 'Rosenbrock', 'Ackley')")
            exit(1)

    def __calculate(self, index: int) -> float:
        # print(f"log({self.num_selection} + {1}) - log({index + 1}) = {(np.log(self.num_selection +1) - np.log(index+1))}")
        return  (np.log(self.num_selection +1) - np.log(index+1))
    
    def __initparam(self) -> np.array:
        weights = [self.__calculate(index) for index in range(self.num_selection)]
        weights = weights / sum(weights)
        sigma = np.identity(self.num_parameters) * self.init_singma
        mu = np.zeros(self.num_parameters)
        return np.array(weights), sigma, mu
    
    def __sample(self, mu:np.array , sigma: np.array) -> np.array:
        # print(mu, sigma)
        pop = np.random.multivariate_normal(mean = mu, cov = sigma, size = self.num_individuals)
        fitness = [self._objectsfunction(ind) for ind in pop]
        return pop, np.array(fitness)

    def get_pop(self, pop: np.array, fitness: np.array) -> np.array:
        argsort = np.argsort(fitness)
        pop_select = [pop[index] for index in argsort[ : self.num_selection]]
        return np.array(pop_select)

    def __update(self, pop_select: np.array, sigma: np.array, mu:np.array, weights: np.array) -> np.array:
        sigma_update, mu_update = np.zeros((self.num_parameters, self.num_parameters)), np.zeros(self.num_parameters)
        # print(sigma_update, mu_update)
        for index, value in enumerate(pop_select):
            mu_update += weights[index] * value
            sigma_update += weights[index] * np.matmul((value - mu), (value - mu).T)
        sigma_update += self.initi_epsilon * np.identity(self.num_parameters)

        return sigma_update, mu_update

    def sover(self, path_file_gif: str, path_file_logger: str):
        np.random.seed(self.random_seed) 
        GEN = 0
        weights, sigma, mu = self.__initparam()
        pop, fitness =self.__sample(mu, sigma)
        best_ind = np.argmin(fitness)
        best = fitness[best_ind]
        num_of_eval = self.num_individuals
        pop_select = self.get_pop(pop, fitness)
        sigma, mu = self.__update(pop_select, sigma, mu, weights)

        history_pop = [pop]
        history_fitness = [fitness]
        history_best_eval = [[num_of_eval, best]]

        file_logger = os.path.join(path_file_logger, f"CEM_{self.objective_function}_{self.num_parameters}_{self.num_individuals}_{self.random_seed}.txt")
        f = open(file_logger, 'w+')
        f.write("#GEN\tx_best\tf(x_best)\t#eval\n")
        f.write(f"{ GEN }\t{ pop[best_ind] }\t{ best }\t{ num_of_eval }\n")

        print(f"#GEN: {GEN}\n\tBest_ind: {pop[best_ind]}\tfitness: {best}")
        print(f"\tsigma = {sigma}, mu = {mu}")

        while True:

            pop, fitness =self.__sample(mu, sigma)
            best_ind = np.argmin(fitness)
            best = fitness[best_ind]
            pop_select = self.get_pop(pop, fitness)
            GEN += 1
            num_of_eval += self.num_individuals
            #update mu, sigma
            sigma, mu = self.__update(pop_select, sigma, mu, weights)
            history_pop.append(pop)
            history_fitness.append(fitness)
            history_best_eval.append([num_of_eval, best])
            # save info
            f.write(f"{ GEN }\t{ pop[best_ind] }\t{ best }\t{ num_of_eval }\n")

            print(f"#GEN: {GEN}\n\tBest_ind: {pop[best_ind]}\tfitness: {best}")
            print(f"\tsigma = {sigma}, mu = {mu}")

            if num_of_eval >= self.max_of_eval:
                break
        #save result end 
        f.write(f"Result: mean = {np.mean(np.array(history_best_eval)[:,1])} std = {np.std(np.array(history_best_eval)[:,1])}")
        f.close()

        history_pop, history_fitness = np.array(history_pop), np.array(history_fitness)
        # if self.num_parameters == 2 and self.num_individuals == 32 and self.random_seed == 20521888:
        #     file_name = os.path.join(path_file_gif,f"CEM_{self.objective_function}_{self.num_parameters}")
        #     draw_graph3D(xdata = history_pop[:,:,0] ,
        #          ydata = history_pop[:,:,1],
        #          zdata = history_fitness,
        #          objective_function = self.objective_function,
        #          filename = file_name)
        
        return history_best_eval

if __name__ == "__main__":
    # algorithm = DE(objective_function = 'Michalewicz')
    # algorithm.sover("./",'./')
    cem =CEM(objective_function = "Ackley")
    cem.sover("./",'./')
