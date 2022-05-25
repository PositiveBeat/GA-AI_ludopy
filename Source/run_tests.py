'''
Multiprocessing source: https://machinelearningmastery.com/multiprocessing-in-python/
'''

import copy
import multiprocessing as mp
import numpy as np
import time

from game_manager import run_game
from GA_ai import Network, Population
from logger import Logger


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '■' * int(percent/2) + '-' * (50 - int(percent/2))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


if __name__ == '__main__':
    pool = mp.Pool(10)  # Max processes running at a time
    start_time = time.perf_counter()

    # Logging information
    log = Logger('plot')
    log.log_to_file('fitness')   # Print to log
    log_chromosome = Logger('Chromosome_database')
    log_chromosome.log_to_file('Code')
    avg_fitness = 0


    gen_limit = 1
    pop_size = 40
    game_quantity = 50


    # Initialize network (input_size, output_size, nr_hidden_layers, hidden_size)
    network = Network(44, 4, 2, 10)
    print('Network element size: ' + str(network.total_nr_elements) + '\n')
    # Initialize population (population_size, code_size)
    pop = Population(pop_size, network.total_nr_elements)

    # pop.overwrite_population([-0.7910063510336723, 2.2085666441266443, 3.4486242942497487, 2.194350186190431, -0.9462906291398927, 0.0, 0.4702896152817513, -1.2336207092495135, 1.6827192299670686, 0.801479648545496, 1.7266019900951448, 0.28539533059806926, 0.45966219834390243, -0.6765072917314993, -2.7701841632012725, 1.0377173125722274, 0.14752235886931395, 1.4113296192567377, 0.0, 1.0681610337342475, 1.358396225413673, 1.0569496370767548, -0.6795769917793829, -0.9011577288093119, 2.1701530499396147, -0.9669356102626674, 4.296301359506968, -0.29454015688509017, 4.342894807740707, 3.087593858293503, 0.37165994585148576, -1.088450985277787, -0.3141315888477023, 0.0, 0.2516652756078632, 0.9296061795931609, 3.218171822482166, 0.0, 1.0355983324503608, 0.28950520647799843, -4.389797573147646, -0.9852484610124891, -1.0102975070657492, -1.6095008566244895, 2.156696044826565, 1.8847329367960304, -1.790988511797444, -0.668373364252596, 0.4027379220409822, -1.886649122920935, -0.023330730391877826, 0.19944656818101336, -1.7513364712673771, 2.905328344811046, -0.7916926209948713, 1.539088162904674, -1.7668686715045174, -3.9466182805071384, -1.8468516817639318, 1.4928874183991803, 3.027893442582443, 0.35856973154936617, 3.7186001563587343, 3.2011476352467145, -1.8167697622756256, 2.834327645077728, 1.2484450808995016, -0.21540959843566732, -2.5153435834892397, 0.0, -0.25946888715879185, 1.8069743745867524, 0.9673216306976415, -2.8444454696074244, 2.6394946594268793, 0.7966256490916317, 0.8610360745000359, -1.7704185889025048, 0.15537546424491788, -0.8948238492948737, -2.6158990731495955, 0.5657389339131645, -2.0219904050726765, 3.7329719124741714, -1.2953891369268935, 0.5262348673618069, 0.5320169534945844, 0.0, -0.8906298080376708, 0.22517219356000884, 0.3808996071704638, 0.0, 0.7573891232044707, 3.624474553264647, 0.6085540032337261, 1.8157945223483467, 0.001780156044520087, -1.5358397188521282, 1.3520301605615355, -0.9343599892049563, 0.18547725687665206, 1.3776044290444727, -2.0087505814484476, -1.1918854599179545, 0.7377555583396476, 0.09732263890003123, 0.9906903792534337, -0.6101694394303996, 0.7460474089615767, 0.21568981733106335, 0.2771610969459537, 1.9415756711228922, 0.39433236761828194, 0.1455242468415341, -0.565861677971208, 1.2332839254447499, 3.8940273487713033, -1.4540576571587467, 1.630166952101529, -1.1811449510072687, -0.9289603898701344, -1.3287958443406116, 0.6067400961055003, 0.4141176969195217, -0.1283555719644725, 0.2275580007435747, -3.434750832636243, -0.5389012349227087, -0.6605634289763527, -0.32077872286186326, -1.0363212077848847, -0.3906026757895509, 1.250551252202173, -0.8010168601612313, 4.340547436456774, 1.0987335605102748, -2.618532674511225, -2.870083450072735, 0.5242700300478201, 0.9700061227946413, -1.108363644927714, 0.8643752443853996, -0.5224188691828219, -1.1492863616470375, -0.4867036333209264, 2.2837105557668744, 2.3834076573883425, 0.3612180026594823, 0.0, 0.0, -0.03362964449061172, 0.25937913790369543, 0.05096380532714378, -0.023328258714598947, -3.1989177400822517, 0.0, -0.6970423372983409, -1.8402609967832892, 1.2492363981769539, 0.27830044827617745, 0.35900664702085683, -0.75691038673464, 0.6776462394467817, -0.3018815083452746, -1.014761807472131, 0.0, -0.23009572441323747, -2.862245247719845, -0.052337117413883805, -0.6793745377825631, -1.6254888193169261, 0.27767923152163165, -0.8292431280826923, -1.8913429481410862, -1.1506040439344352, -0.32579151682928886, 0.13672638896931022, 2.562644558130529, -0.3285288065788056, 2.0519558064122667, -0.5426063943716576, -0.4110283116138932, -3.5072892761465884, -1.5637209210604006, -4.060230835915279, -1.4042005552607255, 1.5931818793952863, 2.5481310185040247, -2.1735993961016007, 1.0017392542398234, -0.5716577058059621, -1.0755379742449869, 0.32861528660952555, 1.505989202344703, 0.021593826914172397, -1.4367585091477109, 0.04795015478393478, 1.90813698269716, -0.8974607226370568, -1.3569369677421403, 0.5700021908616258, 0.6439644482034395, -1.8790149130576337, -2.171263359745731, 1.205379314018517, -1.2497177245540703, 0.9240692159376882, 1.66737325196075, -1.3481298477407442, -1.112680794684556, -3.0345346714600927, -0.8507776794135599, -0.3786462755851723, -0.181615451285432, 0.0, -0.15688111677193795, 0.6877926725843104, -1.7428465978033703, -0.18762258556311578, 1.6285878376437861, -0.4291246411886575, 0.7054751299941413, 0.0, -1.0673592287194027, 0.0, -3.19260496931079, 0.0, -0.9730094724711411, -0.1210354567943368, 0.0, -0.4951692799546143, -2.9084307073804077, -0.5351843223927207, 1.6836134790682489, 0.10091190291106543, 1.979825272647644, 1.3836044761390796, -1.5233628327566495, 1.9892599894223335, 1.5315169547272196, -0.5858106693940381, -0.19278380336581777, 0.0, 1.643444099249206, -0.8244426019403626, 0.791495894089806, -2.5660576139800604, 0.5993013345965522, -2.0311844798493333, -2.281689671273582, -3.2651888138722462, -1.194074135935161, -0.08366870062074883, -4.029282363522776, 0.958091134843404, -0.8434186506908585, -0.7343027905334842, -0.3122787563372775, -0.7703228187060127, 2.0015495773004415, 1.8063250586438866, 1.5934315327331798, -2.383301573156317, 0.0, -0.1730056889342918, 2.0856807908354407, 1.7635062085952582, 0.2532800168405865, 0.5863918250064925, -0.9820957057874206, -0.062498467234641863, 0.21146216325971684, 2.283851284059793, -1.2694538018830241, 2.251417841708877, -1.2828073414801495, -1.410808094981968, -1.4051249339221104, -3.54147412801062, -2.8656897934771792, -3.870072746352211, -0.623985915411523, -1.0380562804640852, -0.8866895736554667, -0.8890133302603908, 0.3047656411350856, -2.4429219311654164, -2.4777291525195495, 0.6708839203074753, 0.0, -0.9569824773262071, 0.14538976869765174, -1.4076774232444287, -1.514910438465093, 1.4810137823977043, 0.5487841719518711, -1.3886891492668765, -1.580750871332005, 4.6625973143322526, -0.6522207206339339, 2.680844938183777, -1.191677568411587, 0.9035493065018663, -3.6033448701187463, -1.4828910710683996, 0.9177017962522522, 1.5376745483936918, -0.09185305232326124, -1.813951850887174])


    for generation in range(gen_limit):
        progress_bar(generation, gen_limit)
        
        avg_fitness = 0
        games_won = 0
        
        for chromosome in pop.population:
            network.update_layers(chromosome.code)

            # Spawn processes to run games in parallel
            processes = [pool.apply_async(run_game, args=(copy.deepcopy(network),)) for _ in range(game_quantity)]
            result = np.array([p.get() for p in processes])
            
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
        child = pop.recombination(parent1, parent2)
        
        pop.overwrite_population(child.code)
        log_chromosome.log_to_file(generation, child.code.tolist())
        pop.mutate(0.001)
        

    finish_time = time.perf_counter()
    print(f"\nProgram finished in {finish_time-start_time} seconds")
