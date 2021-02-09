import numpy as np
from pyit2fls import  IT2FS_plot, IT2FS_Gaussian_UncertMean
from numpy import linspace


def get_best(population, fitness_func, best, fbest):
    for i in range(population[0].shape[0]):

        if population[1][i] > -1.0:
            tmp = population[1][i]
        else:
            tmp = fitness_func(population[0][i])
            population[1][i] = tmp

        if best is None or tmp < fbest:
            best = population[0][i]
            fbest = tmp

    return best.copy(), fbest.copy()


def tournament_pair(population, fitness_func):
    idxs = np.random.permutation(np.arange(population[0].shape[0]))

    parent1 = population[0][idxs[0], :]
    parent2 = population[0][idxs[1], :]

    if population[1][idxs[0]] > -1.0:
        fitness1 = population[1][idxs[0]]
    else:
        fitness1 = fitness_func(parent1)
        population[1][idxs[0]] = fitness1

    if population[1][idxs[1]] > -1.0:
        fitness2 = population[1][idxs[1]]
    else:
        fitness2 = fitness_func(parent2)
        population[1][idxs[1]] = fitness2

    return parent1 if fitness1 < fitness2 else parent2


def generate_individual(size):
    individuals = np.zeros(size)
    individuals[0] = np.random.uniform(low=0.0, high=1.)
    individuals[1] = np.random.uniform(low=0.0, high=0.5)
    individuals[2] = np.random.uniform(low=0.0, high=0.25)
    individuals[3] = np.random.uniform(low=0.0, high=1.)
    individuals[4] = np.random.uniform(low=0.0, high=0.5)
    individuals[5] = np.random.uniform(low=0.0, high=0.25)
    individuals[6] = np.random.uniform(low=0.0, high=1.)
    individuals[7] = np.random.uniform(low=0.0, high=0.5)
    individuals[8] = np.random.uniform(low=0.0, high=0.25)

    return individuals


def mutation(individual):
    idx = np.random.randint(low=0, high=individual.shape[0])
    new = generate_individual(9)
    individual[idx] = new[idx]
    return individual


def _crossover(male, female):
    index1 = np.random.randint(low=1, high=male.shape[0])
    index2 = np.random.randint(low=1, high=male.shape[0])
    if index1 > index2: index1, index2 = index2, index1
    child1 = []
    child2 = []
    child1[0:index1] = male[0:index1]
    child1[index1:index2] = female[index1:index2]
    child1[index2:] = male[index2:]
    child2[0:index1] = female[0:index1]
    child2[index1:index2] = male[index1:index2]
    child2[index2:] = female[index2:]

    return child1, child2


def genetic_algorithm(fitness_func, genotype_size, generation_size=30, generations=20, crossover_rate=0.9, mutation_rate=0.2,
                      debug=False):
    best_so_far, best_fitness = None, None
    population = [np.array([generate_individual(genotype_size) for _ in range(generation_size)]),
                  np.zeros(generation_size) - 1.0]
    next_gen = np.zeros((generation_size, genotype_size))

    for generation in range(generations):
        for pairs in range(0, generation_size, 2):

            parent1 = tournament_pair(population, fitness_func)
            parent2 = tournament_pair(population, fitness_func)

            while np.array_equal(parent1, parent2):
                parent2 = tournament_pair(population, fitness_func)

            if np.random.uniform() < crossover_rate:
                offspring1, offspring2 = _crossover(parent1, parent2)
                next_gen[pairs, :] = offspring1
                next_gen[pairs + 1, :] = offspring2
            else:
                next_gen[pairs, :] = parent1
                next_gen[pairs + 1, :] = parent2

            if np.random.uniform() < mutation_rate:
                next_gen[pairs, :] = mutation(next_gen[pairs, :])

            if np.random.uniform() < mutation_rate:
                next_gen[pairs + 1, :] = mutation(next_gen[pairs + 1, :])

        best_so_far, best_fitness = get_best(population, fitness_func, best_so_far, best_fitness)

        population[0][:] = next_gen[:]
        population[1][:] = -1.0
        next_gen[:] = 0.0

        if debug:
            print(best_so_far)
            print('Generation {:2d}, best fitness = {:.10f}'.format(generation, best_fitness))

            domain = linspace(0.0, 1., 100)
            Short = IT2FS_Gaussian_UncertMean(domain, [best_so_far[0], best_so_far[1], best_so_far[2], 1.])
            Medium = IT2FS_Gaussian_UncertMean(domain, [best_so_far[3], best_so_far[4], best_so_far[5], 1.])
            Long = IT2FS_Gaussian_UncertMean(domain, [best_so_far[6], best_so_far[7], best_so_far[8], 1.])
            IT2FS_plot(Short, Medium, Long, title=generation, legends=["Short", "Medium", "Long"])

    return best_so_far, best_fitness
