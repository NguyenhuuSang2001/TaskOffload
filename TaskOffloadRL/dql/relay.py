"""
Memory Mechanism Implementation
@Authors: TamNV
"""
import random

class MemoryBase(object):
    # Define Abstract Class for MemmoryBase
    def __init__(self, capability=100000):
        """
        Initialize method for creating a new instance of memory based
        + Params: capability: Integer
        """
        self.capability = capability
        self.queue = {
                        "s":[],
                        "a":[],
                        "ns":[],
                        "r":[],
                        "d":[]
                    }
        self.cur_size = 0
        self.pris = None

    def insert_samples(self, samples):
        """
        Perform Inserting New Observations into Memory
        + Params: samples
        """
        pass

    def sel_samples(self, batch_size):
        """
        Perform Selecting Observations from Previous
        """
        pass

    def clrqueue(self):
        """
        Perform Clearing Observations in this Queue
        """
        pass

class Memory(MemoryBase):
    """
    Declare Memory for Training Deep Reinforcement Learning
    """
    def __init__(self, capability):
        super(Memory, self).__init__(capability)

    
    def insert_samples(self, samples):
        """
        Insert Samples to Queue
        + Params: samples: Dictionary
        + Returns: None 
        """
        if self.queue.keys() != samples.keys():
            raise Exception("Inserted Samples are't the same format")
        num_sams = len(samples["s"])
        for key in self.queue.keys():
            self.queue[key] = samples[key] + self.queue[key]
        if self.cur_size + num_sams <= self.capability:
            self.cur_size += num_sams
            return 
        # Remove Over Samples in a Queue
        for key in self.queue.keys():
            self.queue[key] = self.queue[key][0:self.capability]
        self.cur_size = self.capability
        return

    def sel_samples(self, batch_size):
        """
        Select n samples from queue
        + Params: batch_size: Integer
        """
        batch = {"s":[], "a":[], "ns":[], "r":[], "d":[]}
        
        if batch.keys() != self.queue.keys():
            raise Exception("Format of Batch and Queue must be same")
        if self.cur_size <= batch_size:
            return self.queue 
        idxs = random.sample(list(range(self.cur_size)), batch_size)
        for idx in idxs:
            for key in self.queue.keys():
                batch[key].append(self.queue[key][idx])
        return batch

    def clrqueue(self):
        """
        Perform Clearing Observations
        """
        self.cur_size = 0
        for key in self.queue.keys():
            self.queue[key] = []
