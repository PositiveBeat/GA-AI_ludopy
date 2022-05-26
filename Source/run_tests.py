'''
Multiprocessing source: https://machinelearningmastery.com/multiprocessing-in-python/
Network rules of thumb: https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af
'''

import copy
import multiprocessing as mp
import numpy as np
import time

from game_manager import run_game, init_pool_processes
from GA_ai import Network, Population
from logger import Logger


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '■' * int(percent/2) + '-' * (50 - int(percent/2))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


if __name__ == '__main__':
    pool = mp.Pool(10, initializer=init_pool_processes)  # Max processes running at a time
    start_time = time.perf_counter()

    # Logging information
    log = Logger('plot')
    log.log_to_file('fitness')   # Print to log
    log_chromosome = Logger('Chromosome_database')
    log_chromosome.log_to_file('Code')
    avg_fitness = 0


    gen_limit = 100
    pop_size = 40
    game_quantity = 50


    # Initialize network (input_size, output_size, nr_hidden_layers, hidden_size)
    network = Network(48, 4, 3, 20)
    print('Network element size: ' + str(network.total_nr_elements) + '\n')
    # Initialize population (population_size, code_size)
    pop = Population(pop_size, network.total_nr_elements)
    pop.mutate(0.5)
    
    # pop.overwrite_population([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4053748340450252, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5611157224948267, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01787428007138782, 0.0, 0.0, -1.9214076906963642, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0286195122964485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6300919800124518, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.21873171255811338, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15288075175159996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2812665189400319, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.011918689493970454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10661620947178636, 0.0, -1.5265159511398518, 0.0, 0.0, 0.7592096288974584, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3525613023863197, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3861084027245738, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2924157453247899, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.612176880714742, -0.6125842512816538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1088166429967157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    for generation in range(gen_limit):
        progress_bar(generation, gen_limit)
        
        avg_fitness = 0
        games_won = 0
        
        for chromosome in pop.population:
            network.update_layers(chromosome.code)

            # Spawn processes to run games in parallel     
                   
            # Not working multiprocessing:
            # processes = []
            # with mp.Pool(mp.cpu_count()-2 or 1, initializer=init_pool_processes) as pool:
            #     processes = [pool.map(run_game, (copy.deepcopy(network),)) for _ in range(game_quantity)]
            # result = np.array(processes)
            
            # result = [pool.map(run_game, (copy.deepcopy(network),)) for _ in range(game_quantity)]
            
            # Working multiprocessing:
            processes = [pool.apply_async(run_game, args=(copy.deepcopy(network),)) for _ in range(game_quantity)]
            result = np.array([p.get() for p in processes])
            
            # No multiprocessing:
            # result = []
            # for _ in range(game_quantity):
            #     result.append(run_game(copy.deepcopy(network)))
            # result = np.array(result)

            score = np.sum(result)
            chromosome.fitness = score
            avg_fitness += score
            games_won += np.count_nonzero(result == 3)
        
        # logging
        avg_fitness /= pop_size
        games_won /= pop_size
        games_won_percent = 100 * games_won / game_quantity
        log.log_to_file(generation, games_won_percent, avg_fitness)
        
        # Update population
        parent1, parent2 = pop.choose_parents_tournament(k = int(np.ceil(pop_size/8)))
        child = pop.recombination(copy.deepcopy(parent1), copy.deepcopy(parent2))
        pop.new_generation(copy.deepcopy(parent1), copy.deepcopy(parent2), copy.deepcopy(child))
        
        log_chromosome.log_to_file(generation, child.code.tolist())
        pop.mutate(0.01)

        

    finish_time = time.perf_counter()
    print(f"\nProgram finished in {finish_time-start_time} seconds")
