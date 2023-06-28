# https://towardsdatascience.com/using-the-right-dimensions-for-your-neural-network-2d864824d0df

import copy
import numpy as np
import random

class Layer:
    
    def __init__(self, nr_inputs, nr_elements):
        self.w = np.zeros((nr_elements, nr_inputs))
        self.b = np.zeros((nr_elements, 1))
        
        self.w_n = nr_inputs * nr_elements
        self.b_n = nr_elements
        
    def __activation(self, z, func):
        if (func == 'linear'):
            return z
        if (func == 'sigmoid'):
            return 1 / (1 + np.exp(-z)) 
        
    def compute_layer(self, input, activation_func='linear'):
        Z = self.w @ input + self.b
        a = self.__activation(Z, activation_func)
        return a
        
    def update_weights(self, weights, biases):
        self.w = copy.deepcopy(weights)
        self.b = copy.deepcopy(biases)


class Network:
    
    def __init__(self, nr_inputs, nr_outputs, nr_hidden_layers, layer_size):

        # Initialise network architecture
        self.hidden_layers = np.array([Layer(nr_inputs, layer_size)])
        if (nr_hidden_layers > 1):
            for _ in range(nr_hidden_layers - 1):
                self.hidden_layers = np.append(self.hidden_layers, Layer(layer_size, layer_size))

        self.output_layer = Layer(layer_size, nr_outputs)
        
        # Count number of elements in network
        sum = 0
        for layer in self.hidden_layers:
            sum += layer.w_n + layer.b_n
        sum += self.output_layer.w_n + self.output_layer.b_n
        self.total_nr_elements = sum
        
        
    def __propagate_network(self, input):
        propergated_input = input
        for layer in self.hidden_layers:
            propergated_input = layer.compute_layer(propergated_input)
        
        return propergated_input
    
    
    def compute_result(self, input):
        network_result = self.__propagate_network(input)            # Hidden layers
        result = self.output_layer.compute_layer(network_result)    # Final result
        
        if (len(result) == 1):
            result = float(result[0])
        
        return result
        
        
    def update_layers(self, code):  # Decoding GA chromosome
        
        if (len(code) != self.total_nr_elements):
            raise RuntimeError(f'Code length does not match network architecture: {len(code)} != {self.total_nr_elements}')

        # Update hidden layers
        offset = 0
        for layer in self.hidden_layers:
            elements = np.array(code[offset:(layer.w_n + layer.b_n)+offset])
            offset += (layer.w_n + layer.b_n)
            weights = np.resize(elements[:layer.w_n], (layer.w.shape[0], layer.w.shape[1]))
            biases = np.resize(elements[layer.w_n:], (layer.w.shape[0], 1))
            layer.update_weights(weights, biases)
            
        # Update output layer
        elements = np.array(code[offset:])
        weights = np.resize(elements[:-self.output_layer.w_n], (self.output_layer.w.shape[0], self.output_layer.w.shape[1])) 
        biases = np.resize(elements[-self.output_layer.b_n:], (self.output_layer.w.shape[0], 1)) 
        self.output_layer.update_weights(weights, biases)







class Chromosome:
    
    def __init__(self, code):
        self.code = np.array(code)
        self.fitness = 0
    
    def __str__(self):
        return 'fitness: ' + str(self.fitness) + ', code: ' + str(self.code)


class Population:
    
    def __init__(self, population_size: int, code_len: int):
        
        # Randomly initialise new population
        self.population = [Chromosome(np.random.uniform(low=0, high=0, size=code_len)) for _ in range(population_size)]
        self.pop_size = population_size
        self.code_len = code_len


    def overwrite_population(self, code):
        for chromosome in self.population:
            chromosome.code = copy.deepcopy(code)
    
    
    def get_fitness(self):
        scores = [c.fitness for c in self.population]
        return scores
    

    def get_best_chromosome(self, population):
        best_chromosome = Chromosome([])
        best_fitness = -1
        
        for chromosome in population:
            if (chromosome.fitness > best_fitness):
                best_fitness = chromosome.fitness
                best_chromosome = copy.deepcopy(chromosome)
        
        return best_chromosome
    
    
    def choose_parents_tournament(self, k):
        pool1 = random.sample(self.population, k)
        pool2 = random.sample(self.population, k)
        
        parent1 = self.get_best_chromosome(pool1)
        parent2 = self.get_best_chromosome(pool2)
        
        return parent1, parent2
    
    
    def recombination(self, parent1, parent2):  # Uniform crossover
        child_code = []
        
        for i in range(0, self.code_len):
            b = random.randint(0, 1)
            if (b == 0):
                child_code.append(copy.deepcopy(parent1.code[i]))
            elif (b == 1):
                child_code.append(copy.deepcopy(parent2.code[i]))
                
        return Chromosome(child_code)
    
    
    def new_generation(self, parent1, parent2, child):
        onethird = int(self.pop_size/3)
        p1_range = range(0, onethird)
        p2_range = range(onethird, 2*onethird)
        
        for i, chromosome in enumerate(self.population):
            if (i in p1_range):
                chromosome.code = copy.deepcopy(parent1.code)
            elif (i in p2_range):
                chromosome.code = copy.deepcopy(parent2.code)
            else:
                chromosome.code = copy.deepcopy(child.code)
       

    def mutate(self, mutation_rate):
        mutation_mu = 0
        mutation_sigma = 1
        
        for chromosome in self.population:
            for i in range(len(chromosome.code)):
                if (random.random() <= mutation_rate):                
                    mutation = np.random.normal(mutation_mu, mutation_sigma)
                    chromosome.code[i] = copy.deepcopy(chromosome.code[i]) + mutation





if __name__ == '__main__':
    
    # Define random input
    x = np.random.randint(10, size=(4, 1))
    
    # Initialize network (input_size, output_size, nr_hidden_layers, hidden_size)
    network = Network(4, 2, 3, 3)
    
    # Initialize population (population_size, code_size)
    pop = Population(10, network.total_nr_elements)
    # pop.overwrite_population([8.792439613138656, -39.880392426874664, 4.185000342498725, 4.040247151804398, 6.412001510136314, -45.168155418603675, -19.025058572574533, 13.415751154975275, 9.341187203398032, -27.05519429056428, -18.924044100673445, 26.884526601523454, 7.307220976434147, -0.20190368364229072, 15.963572866448082, 56.76555435168767, 34.15434587162856, -11.575354877880772, -6.108109766317823, 2.829708024433275, 43.0161526515637, -24.025090358335373, -28.81230875345361, -14.20034630944097, -30.84391779236968, -33.2142280190563, 28.123997080493126, 10.521428646769241, 38.15119525310348, 13.401051584067595, -40.88215712754327, -13.535337675388028, 2.289120699175548, -7.418328664831005, -5.564623415404051, -52.254251855425494, 32.420063332825706, -30.357005064620747, 13.265383273302747, 27.570057205067606, 10.124341103880546, 35.25929351438858, -7.414551869627917, -15.055370138016114, -0.8170712615869913, -25.240277823912997, 40.24530036812092, -25.93372013611277, 62.96987225775262, 38.727368314199246, 34.51651282846604, 17.04356383343552, -47.30361849957171, 10.570023810562164, -41.127169464436896, 27.457176982766367, -22.106696028718016, 9.106158368824579, -17.599460121688843, -6.9964383655834235, 49.086968522596536, 5.251826002068114, 1.1452822770872737, -35.10357308408302, 15.947144769137358, -51.20238110577097, -12.638085366825354, -37.97673841440128, 12.077316066236843, 34.26555647146873, 29.854683807930293, 15.20036792211654, -24.748597170199666, 23.332084918490924, 11.567048320078214, 8.285832484070019, 27.301323059372013, 20.252956709525602, 11.996127953316709, -15.235403484858669, 22.70752799860916, -40.32008430936926, -30.440302038953654, 34.94024235090115, 32.535689284206214, -15.692318826500633, 15.037432738242124, -2.713110276859366, 13.075070656349466, 58.37844311154617, 0.4607597318863692, -7.574892913655006, 43.21553280219598, 43.64815063406041, 49.04734988303803, 38.07225031267161, 8.686416010076524, 12.807968121677524, 8.083559615073602, -9.627830747599283, 27.407422930305934, 5.100768814420284, 4.784073024633065, 6.67667195651455, 0.9527642770447733, -27.353007097651073, -0.4670419050180923, -8.007093703936711, -43.399232580525215, 3.149057935973593, 15.011512953822173, -13.62033081804979, 39.415441821273006, -47.504023404477145, -23.14066402968477, 6.216145726880704, -18.8807615238992, -9.543309250613092, 36.34083696285574, 47.16588517004074, -11.136680593270647, -49.524383794872534, -10.223904674633106, -12.436917951084466, -40.45514256509999, 3.6145787400116216, -13.063459501422157, 4.259218771161124, -48.46032956297912, -18.416279724755306, 31.13816835216423, 4.50559963029664, 13.247449249268588, -5.463411365302856, 43.47612412693609, 16.65545759078932, -15.710022379851909, 0.5648575871573123, -2.6314690950873665, -2.2282098360379874, 39.13621288074336, 1.5670541041087926, 32.279082676879995, 7.545991505586307, 24.368108418854696, 15.481079833029753, 14.989441957429433, -33.24095132958211, -13.58966025424133, 7.173545442321323, -30.26629654382921, -20.040612392970353, 41.27471261389963, -55.41459955019255, 9.324185843123384, -20.548721666402244, -18.76886036315959, -15.68314767242298, -10.873255678199753, -0.21875531407804738, 6.129418370005562, -26.65239436907032, -14.176600160971716, 1.4852691924702315, 26.157164062448444, -28.246430996096976, 22.068906542908096, -2.048729611864889, -21.80875750626153, -6.07895171083076, -2.3081758301628676, 20.036736192758642, -14.197506029386107, 25.32996388110167, 26.41900247654339])
    # for c in pop.population:
    #     print(c)
    # pop.mutate(0.2)
    # for c in pop.population:
    #     print(c)
        
    pop.mutate(0.2)
    p1, p2 = pop.choose_parents_tournament(3)
    child = pop.recombination(p1, p2)
    pop.new_generation(p1, p2, child)
    
    print(p1.code)
    print(p2.code)
    print(child.code)
    print()
    print()
    print()
    
    for c in pop.population:
        print(c)
    

    # code = np.random.randint(2, size=network.total_nr_elements)
    # code = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8])
    # network.update_layers(code)
    
    # print(network.hidden_layers[0].w)
    
    
    # a = network.compute_result(x)
    # print(a)

