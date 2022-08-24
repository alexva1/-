from collections import namedtuple
from functools import partial
from multiprocessing.dummy import Array
from random import choices, randint, random, randrange
from statistics import mean
import string
from typing import Callable, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib import pyplot as plt

dict_length = 8520
max_idf_value = 0
min_idf_value = 0
min_choosen_words = 1000


population_size = 200
mutation_num = 8520
mutation_propability = 0.01
crossover_propability = 0.1

def read_doc(doc_filename: string) -> Array:
    doc = []
    doc_file = open(doc_filename,'r')
    for text in doc_file: 
        word = text.split()
        counter = 1
        sentence_num = int(word[0].strip('<>'))
        vec = np.zeros(dict_length,dtype=int)
        for i in range(sentence_num):
            words_num = int(word[counter].strip('<>'))
            counter += 1
            for j in range(words_num):
                vec[int(word[counter])] += 1
                counter += 1
        doc.append(vec)
    return doc
    
def tf_idf_calc() -> Array: 
    doc1 = np.array(read_doc("Data/train-data.dat"))
    doc2 = np.array(read_doc("Data/test-data.dat"))
    doc = np.concatenate((doc1,doc2),axis=0)
    tf_doc = TfidfTransformer(norm='l2',use_idf=True , smooth_idf= True).fit_transform(doc)
    dense_tf_doc = tf_doc.todense()
    tf_mean = np.zeros(dict_length,dtype=float)
    for i in range(dense_tf_doc.shape[0]):
        for j in range(dense_tf_doc.shape[1]):
            tf_mean[j] += dense_tf_doc[i,j]
    tf_mean = tf_mean/dict_length
    print("idf vec calculated")
    return tf_mean 


Dict = namedtuple('Dict' , ['index','value'])


def get_Dict() -> List[Dict]: 
    dict = []
    dict_idf = []
    tf_idf_vec = tf_idf_calc()
    global max_idf_value
    global min_idf_value
    for i in range(dict_length):
        dict.append(Dict(i,tf_idf_vec[i]))
        dict_idf.append(dict[i].value)
    dict_idf = sorted(dict_idf)
    min_idf_value = mean(dict_idf[0:999])
    max_idf_value = mean(dict_idf[7520:])
    print("dictionary calculated")
    print("min mean : " + str(min_idf_value))
    print("max mean : " + str(max_idf_value))
    return dict



Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome],int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
dict = get_Dict()



def generate_genome() -> Genome: 
    return choices([0,1], k = dict_length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome() for _ in range(size)]

def fitness(genome: Genome) -> float:
    if len(genome) != len(dict):
        raise ValueError("genome and dict must be of the same length")
    
    value = 0
    count_choosen = 0

    for word in dict: 
        if genome[word.index] == 1: 
            value += word.value
            count_choosen += 1
    
    if count_choosen < min_choosen_words: 
        return -10
    
    value = value/count_choosen
    #calculate normalized choosen value
    count_choosen_value = min_idf_value + ((dict_length-count_choosen)*(max_idf_value-min_idf_value))/(dict_length-min_choosen_words)
    
    value += count_choosen_value

    #normalize value to [0,1]
    return (value-min_idf_value)/(2*(max_idf_value-min_idf_value))


def selection_func(population: Population, fitness_func: FitnessFunc) -> Population: 
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def n_point_crossover(a: Genome, b:Genome, n: int, probability = crossover_propability) -> Tuple[Genome,Genome]:
    if random() <= probability:
        if len(a) != len(b):
            raise ValueError("Genome a and b must be of same length")

        length = len(a)
    
        for i in range(n):
            p = randint(1,length-1)
            a = a[0:p] + b[p:]
            b = b[0:p] + a[p:]
    
    return a,b

def mutation(genome: Genome, num: int = mutation_num, probability: float = mutation_propability) -> Genome:
    for _ in range(num): 
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome
    
def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc = fitness,
    selection_func: SelectionFunc = selection_func,
    crossover_func: CrossoverFunc = n_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100,
    improvement_factor: int = 1,
    not_improved_flex: int = 20
) -> Tuple[Population, int]:

    best_solution = []
    generation_num = []
    for k in range(10):

        population = populate_func()

        print('starting model ' + str(k+1))
        print('/n')
        print('/n')
        not_improved_counter = 0
        max_value = 0
        prev_max_value = 0
        mean_value = 0
        min_value = 100

        mean_solution = []
        for i in range(generation_limit):
            population = sorted(
                population,
                key=lambda genome: fitness_func(genome),
                reverse=True
            )

            next_generation = population[0:1] 

            for j in range(int(0.5*(population_size-2))):
                parents = selection_func(population, fitness_func)
                offspring_a, offspring_b = crossover_func(parents[0], parents[1],3)
                offspring_a = mutation_func(offspring_a)
                offspring_b = mutation_func(offspring_b)
                next_generation += [offspring_a, offspring_b]

            prev_max_value = max_value
            prev_mean_value = mean_value

            max_value = 0
            mean_value = 0
            min_value = 100
            for genome in  population:
                val = fitness(genome)
                if max_value < val:
                    max_value = val
                if min_value > val: 
                    min_value = val
                mean_value += val/len(population)
            
            if prev_max_value == max_value:
                not_improved_counter += 1
            else: 
                not_improved_counter = 0

            if not_improved_counter >= not_improved_flex:
                break
            
            

            print("evolution "+str(i+1)+" completed: Mean -> " + str(mean_value) + " Best -> " + str(max_value) + " Worst -> " + str(min_value) + " counter -> " + str(not_improved_counter))
            mean_solution.append(mean_value)
            population = next_generation
        
        plt.plot(mean_solution)
        plt.title('pop_size = 200 , crossover_prob = 0.1 , mutation_prob = 0.01')
        plt.ylabel('finess value')
        plt.xlabel('generation')
        plt.savefig('figures/model'+str(k+1)+'.png')
        plt.clf()
        best_solution.append(max_value)
        generation_num.append(i+1)
        print('Mean number of generations : ' + str(mean(generation_num)))
        print('Mean value of best solution : ' + str(mean(best_solution)))
    return population, i+1

        
population = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(dict)
    ),
    generation_limit=1000,
    improvement_factor=0.0001,
    not_improved_flex=100
)


