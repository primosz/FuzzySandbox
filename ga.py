import numpy as np
from pyit2fls import IT2FS_Gaussian_UncertStd, IT2FS_plot, IT2FS_Gaussian_UncertMean
from numpy import linspace


def _best(population, fitness_func, best, fbest):

    # best, fbest = None, None
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

#comment

def _tournament_selection(population, fitness_func):

    # k == 2
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
    individuals[0] = np.random.uniform(low=0.0, high=0.5)
    individuals[1] = np.random.uniform(low=0.0, high=0.5)
    individuals[2] = np.random.uniform(low=0.0, high=0.25)
    individuals[3] = np.random.uniform(low=0.25, high=0.75)
    individuals[4] = np.random.uniform(low=0.0, high=0.5)
    individuals[5] = np.random.uniform(low=0.0, high=0.25)
    individuals[6] = np.random.uniform(low=0.5, high=1.)
    individuals[7] = np.random.uniform(low=0.0, high=0.5)
    individuals[8] = np.random.uniform(low=0.0, high=0.25)
 #   individuals = np.random.rand(size)
 #   for x in range(0, individuals.shape[0]):
 #       if x in [2, 5, 8]:
 #           tmp=individuals[x]   #to keep deviation below 0.25
 #           individuals[x] = tmp/4
    return individuals


# Function that mutates an individual
def _mutate(individual):

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
    child1[index1:index2]=female[index1:index2]
    child1[index2:] = male[index2:]
    child2[0:index1] = female[0:index1]
    child2[index1:index2] = male[index1:index2]
    child2[index2:] = female[index2:]

    return child1, child2



def genetic_algorithm(fitness_func, dim, n_individuals=30, epochs=20, crossover_rate=0.9, mutation_rate=0.2, verbose=False):

    assert n_individuals % 2 == 0

    population = [np.array([generate_individual(dim) for _ in range(n_individuals)]),
                  np.zeros(n_individuals) - 1.0]

    children = np.zeros((n_individuals, dim))

    best, fbest = None, None

    for e in range(epochs):
        for c in range(0, n_individuals, 2):

            parent1 = _tournament_selection(population, fitness_func)
            parent2 = _tournament_selection(population, fitness_func)

            while np.array_equal(parent1, parent2):
                parent2 = _tournament_selection(population, fitness_func)

            if np.random.uniform() < crossover_rate:
                offspring1, offspring2 = _crossover(parent1, parent2)
                children[c, :] = offspring1
                children[c+1, :] = offspring2
            else:
                children[c, :] = parent1
                children[c+1, :] = parent2

            if np.random.uniform() < mutation_rate:
                children[c, :] = _mutate(children[c, :])

            if np.random.uniform() < mutation_rate:
                children[c+1, :] = _mutate(children[c+1, :])

        best, fbest = _best(population, fitness_func, best, fbest)

        population[0][:] = children[:]
        population[1][:] = -1.0
        children[:] = 0.0

        if verbose:
            print('epoch {:2d}, best fitness = {:.10f}'.format(e, fbest))
            print(best)
            domain = linspace(0.0, 1., 100)
            Short = IT2FS_Gaussian_UncertMean(domain, [best[0], best[1], best[2], 1.])
            Medium = IT2FS_Gaussian_UncertMean(domain, [best[3], best[4], best[5], 1.])
            Long = IT2FS_Gaussian_UncertMean(domain, [best[6], best[7], best[8], 1.])
            IT2FS_plot(Short, Medium, Long, title=e,               legends=["Short", "Medium", "Long"])

    return best, fbest