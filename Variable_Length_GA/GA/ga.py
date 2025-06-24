with open("hyperparameters/seedfile.txt", "rb") as f:
        seed_value = int(f.read())

import numpy as np
np.random.seed(seed_value)

from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, fitness_func, pop_size, max_length, elementcells, common, p_new, mutation_factor, K, nbests):
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.max_length = max_length
        self.elementcells = elementcells
        self.common = common
        self.p_new = p_new
        self.mutation_factor = mutation_factor
        self.K = K
        self.nbests = nbests

        self.init_pop()
    
    def init_pop(self):
        self.population = []
        self.fitness_pop = np.zeros(self.pop_size)
        self.training_perf = []
        for j in range(self.pop_size):
            generated_string = np.random.random(self.common + self.elementcells*np.random.randint(1, self.max_length))
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
        length1 = parent1.shape[0]
        length2 = parent2.shape[0]
        possible_splits = list(range(self.elementcells+self.common, max(length1, length2)+1, self.elementcells))
        n_split = np.random.choice(possible_splits)
        unborn1 = np.array(list(parent1)[:min(n_split, length1)] + list(parent2)[min(n_split, length2):])
        unborn2 = np.array(list(parent2)[:min(n_split, length2)] + list(parent1)[min(n_split, length1):])
    
        length1 = min(unborn1.shape[0], self.common + self.elementcells*self.max_length)
        length2 = min(unborn2.shape[0], self.common + self.elementcells*self.max_length)
        child1 = unborn1.copy()[:length1]
        child2 = unborn2.copy()[:length2]
        combine_until = min(length1, length2)
        gamma = np.random.random()
        child1[:combine_until] = gamma*unborn1[:combine_until] + (1-gamma)*unborn2[:combine_until]
        child2[:combine_until] = gamma*unborn2[:combine_until] + (1-gamma)*unborn1[:combine_until]
    
        return child1, child2
    
    def mutation(self, string):
    
        string_length = string.shape[0]
        mutated_string = string.copy()
        
        if np.random.random()<self.p_new:
            if string_length<(self.common + self.elementcells*self.max_length):
                mutated_string = np.zeros(string_length + self.elementcells)
                mutated_string[:string_length] = string
                mutated_string[string_length:] = np.random.random(self.elementcells)
                string_length = mutated_string.shape[0]
        
        if np.random.random()<self.p_new:
            if string_length > self.common + self.elementcells:
                mutated_string = mutated_string.copy()[:(string_length-self.elementcells)]
                string_length = mutated_string.shape[0]

        p_mutation = min(0.99, self.mutation_factor/string_length)
        for k in range(string_length):
            if np.random.random() < p_mutation:
                mutated_string[k] = np.random.random()
    
        return mutated_string
    
    def optimize(self, nsteps):
        for nstep in tqdm(range(nsteps)):
            self.GA_step()
            self.training_perf += [np.max(self.fitness_pop)]
    
    def results(self):
        return self.population[np.argmax(self.fitness_pop)], self.training_perf
        