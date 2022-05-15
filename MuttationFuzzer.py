
import random
import numpy as np
from train import load_data, load_model
import os
from utils import embed_test_data

class MutationFuzzer():

    def __init__(self,embeddings,iterations,changes):
        super().__init__()
        self.test_emmbeddings = embeddings
        self.mutators = [self.flip,self.replace,self.replace_sample]
        self.iterations = iterations
        self.num_chanages = changes
        self.mean = np.mean(self.test_emmbeddings)
        cov = np.std(self.test_emmbeddings)
        self.std = cov*cov
        self.min = np.min(np.min(self.test_emmbeddings))
        self.max = np.max(np.max(self.test_emmbeddings))

    

    def replace(self,seed):
       
        for i in range(self.num_chanages):
            replace_indx = random.randint(0,len(seed)-1)
            replace_number = random.uniform(self.min,self.max)
            seed[replace_indx] = replace_number
        
        return seed


    def flip(self,seed):

        for i in range(self.num_chanages):
            replace_indx = random.randint(0,len(seed)-1)
            seed[replace_indx] = -1 * seed[replace_indx]

        return seed

    def replace_sample(self,seed):

        for i in range(self.num_chanages):
            replace_indx = random.randint(0,len(seed)-1)
            replace_number = np.random.normal(self.mean,self.std)
            seed[replace_indx] = replace_number
        
        return seed

        

       
    
    def mutate(self,fuzzed_embeddings,indices):
        num_sample = 0

        if (len(indices) == 0):
            num_sample = 10
            indices = random.sample(range(0, self.test_emmbeddings.shape[0]),num_sample)
        else:
            num_sample = len(indices)
        
        if fuzzed_embeddings is None:
            fuzzed_embeddings = self.test_emmbeddings[indices]

        for i in range(self.iterations):
            for fuzzed_embedding in fuzzed_embeddings:
                mutator = random.choice(self.mutators)
                fuzzed_embedding = mutator(fuzzed_embedding)

        return fuzzed_embeddings

if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(),'model')
    model = load_model(model_path)

    mutation_fuzzer = MutationFuzzer(model,10,5)
    seeds = mutation_fuzzer.mutate(mutation_fuzzer.test_emmbeddings[[1,2,3,4,5,6,7,8,9,0]],[1,2,3,4,5,6,7,8,9,0])




        







    




