import random
import numpy as np
from time import time
import math
import os

current_path = os.path.dirname(__file__)


seed = 6

class SystemSimulator():

    def __init__(self, origin_destination_matrix: np.array):
        
        self.origin_destination_matrix = origin_destination_matrix
        self.N_time_frames, self.N_zones, _ = self.origin_destination_matrix.shape
        self.time_frame_flows = {tf: list() for tf in range(self.N_time_frames)}
        self.__simulate_system()
        # print(f'time_frame_flows: \n {self.time_frame_flows}')
        return

    def __simulate_system(self):

        for t in range(self.N_time_frames):
            od_matrix = np.copy(self.origin_destination_matrix[t])
            trips = [[] for _ in range(self.N_zones)]
            for i in range(self.N_zones):
                for j in range(self.N_zones):
                    d = od_matrix[i, j]
                    if d > 0:
                        for _ in range(d):
                            trips[i].append((i,j))
                self.time_frame_flows[t].extend(trips[i])
            np.random.seed(seed)
            np.random.shuffle(self.time_frame_flows[t])
        return

    def get_flow_by_time_frame(self, time_frame: int):
        return self.time_frame_flows[time_frame]
    

if __name__ == '__main__':
    N_zones = 4
    N_timeframes = 30

    # for n in range(5):
    #     SystemGeneration(N_zones,N_timeframes).generate_distances(n)

    # SystemSimulator(np.load(current_path + '/od_mat.npy').astype(np.uint8))