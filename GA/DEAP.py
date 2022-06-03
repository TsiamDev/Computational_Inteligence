# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:25:19 2022

@author: Konstantinos Tsiamitros
"""

# based on examples from thisa site:
#https://deap.readthedocs.io/en/master/examples/ga_onemax.html

import random

import math

import pprint
import bitarray

from deap import base
from deap import creator
from deap import tools

from constants import *
import load_data
#from GA import PrepareFitnessData

def get_tf(words): 
    doc_tfs = []
    total_doc_terms = []
    idf_denominator = dict()
    for doc in words:
        if doc:
            doc_dict = dict()
            cnt = 0
            for sent in doc:
                for s in sent:
                    cnt = cnt + 1
                    if s not in doc_dict:
                        doc_dict[s] = 0
                    doc_dict[s] = doc_dict[s] + 1
            
            doc_tfs.append(doc_dict)
            total_doc_terms.append(cnt)
            
            doc_set = set(doc_dict)
            for t in doc_set:
                if t not in idf_denominator:
                    idf_denominator[t] = 1
                else:
                    idf_denominator[t] = idf_denominator[t] + 1

    # len(words) : is the number of documents
    return doc_tfs, total_doc_terms, len(words), idf_denominator

def get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator):
    # key: term, value: present in # of docs
    term_dict = dict()
    idfs = dict()
    # for every document
    for i in range(0, len(doc_tfs)):
        # for every term
        for term in doc_tfs[i]:
            tf_temp = doc_tfs[i][term] / total_doc_terms[i]
            idf_temp = math.log(num_of_docs/idf_denominator[i])
            if term not in term_dict:
                term_dict[term] = 0
            term_dict[term] = term_dict[term] + 1
            if term not in idfs:
                idfs[term] = 0
            idfs[term] = tf_temp * idf_temp + idfs[term]
            
    # get the means
    mean_idfs = dict()
    for k,v in term_dict.items():
        mean_idfs[k] = idfs[k] / term_dict[k]        
    
    # i think returning just idfs is enough (?)
    return term_dict, idfs, mean_idfs

def PrepareFitnessData():
    #data = load_raw_data.preprocess_vecs()
    file = 'C:\\Users\\HomeTheater\\Desktop\\GA\\Dataset\\Data\\train-data.dat'
    glob_words, glob_sent_num, glob_word_num = load_data.read_file(file)
    
    doc_tfs, total_doc_terms, num_of_docs, idf_denominator = get_tf(glob_words)
    
    tfs, idfs, mean_idfs = get_tfs_idfs(doc_tfs, total_doc_terms, num_of_docs, idf_denominator)
    
    #plt.plot(range(0, len(doc_tfs[0])), doc_tfs[0].values())
    #m_idfs = sorted(mean_idfs, reverse=True)
    #plt.plot(range(0, len(m_idfs)), m_idfs)
    #plt.plot(range(0, len(idfs)), idfs)
    #plt.xlabel('position of term in dictionary')
    #plt.ylabel('mean - inverse document frequency')
    #plt.title('TF-IDF')
    #plt.show()
    
    return mean_idfs

def Evaluate(individual, mean_idfs):
    ones = []
    min_idfs = min(mean_idfs)
    # Calculate the scores for each pop
    #print(type(individual))
    #pprint.pprint(bitarray.bitarray(individual[0]))
    cnt = sum(individual)
    print(cnt)
    print(mean_idfs[cnt])
    ones.append(cnt)
    score = mean_idfs[cnt]
    #print(cnt)
    # reject if pop has less 1's than allowed
    #if cnt < ONES_THRESH:
    #    #print("reject")
    #    score = (0.01,)
    # apply penalty if has more 1's than allowed
    #elif cnt > ONES_UPPER_THRESH:
    #    #print("penalty")
    #    score = mean_idfs[cnt] - min_idfs
        
    return score



def main():
    print("Preparing Fitness Data...")
    mean_idfs = PrepareFitnessData()
    print("Ready!")
    
    print("Initializing...")
    #kargs = {}
    #kargs['weights']=1.0
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    # Attribute generator 
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("mean_idfs", list, mean_idfs)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 8520)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", Evaluate, toolbox.mean_idfs)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    

    
    pop = toolbox.population(n=300)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    # Variable keeping track of the number of generations
    g = 0

    fits = [1]
    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind, mean_idfs)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
    
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
    
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
main()
    