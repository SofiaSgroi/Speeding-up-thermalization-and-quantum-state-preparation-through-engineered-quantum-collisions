with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, fitness_func, pop_size, length, elementcells, mutation_factor, K, nbests):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.length = length
        self.elementcells = elementcells
        self.p_mutation = mutation_factor/float(self.elementcells*self.length)
        self.K = K
        self.nbests = nbests

        self.init_pop()
    
    def init_pop(self):
        self.population = []
        self.fitness_pop = np.zeros(self.pop_size)
        self.training_perf = []
        for j in range(self.pop_size):
            generated_string = np.random.random(self.elementcells*self.length)
            self.population += [generated_string]
            self.fitness_pop[j] = self.fitness_func(generated_string)

    def GA_step(self):
        max_indx = np.argsort(self.fitness_pop)[::-1][:self.nbests]
        new_population = [self.population[indx] for indx in max_indx]
        new_fitness_pop = np.zeros(self.pop_size)
        new_fitness_pop[:self.nbests] = self.fitness_pop[max_indx]

        for j in range(self.nbests, self.pop_size, 2):
            p1 = self.tournament()
            p2 = self.tournament()
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutation(c1)
            c2 = self.mutation(c2)
        
            new_population += [c1]
            new_population += [c2]
            new_fitness_pop[j] = self.fitness_func(c1)
            new_fitness_pop[j+1] = self.fitness_func(c2)
        
        self.population =  new_population
        self.fitness_pop = new_fitness_pop

    def tournament(self):
        best_indx = np.random.randint(self.pop_size)
        best_fit = self.fitness_pop[best_indx]
        for k in range(self.K - 1):
             indx_rnd = np.random.randint(self.pop_size)
             fit_rnd = self.fitness_pop[indx_rnd]
             if fit_rnd > best_fit:
                 best_indx = indx_rnd
                 best_fit = fit_rnd
        return self.population[best_indx]
    
    def crossover(self, parent1, parent2):
        gamma = np.random.random()
        child1 = gamma*parent1 + (1-gamma)*parent2
        child2 = gamma*parent2 + (1-gamma)*parent1
        return child1, child2
    
    def mutation(self, string):
        mutated_string = string
        for k in range(self.elementcells*self.length):
            if np.random.random() < self.p_mutation:
                mutated_string[k] = np.random.random()
        return mutated_string
    
    def optimize(self, nsteps):
        for nstep in tqdm(range(nsteps)):
            self.GA_step()
            self.training_perf += [np.max(self.fitness_pop)]
    
    def results(self):
        return self.population[np.argmax(self.fitness_pop)], self.training_perf
        