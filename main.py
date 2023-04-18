import numpy as np
from plotnine import *
from time import time
import pandas as pd
import json
from DataProcess import *
from SystemSimulator import SystemSimulator
from SystemGeneration import *
from Optimize_relocation import *
from vrp import *
import os

current_path = os.path.dirname(__file__)


class main():
    def __init__(self, paras_relocation, optimization_horizon, look_ahead_horizon, paras_vrp, vrp_model, scenario, case, N_zones=0, coord_depot=[0,0], verbose=False):
        self.paras_relocation = paras_relocation
        self.optimization_horizon = optimization_horizon
        self.look_ahead_horizon = look_ahead_horizon
        self.paras_vrp = paras_vrp
        self.vrp_model = vrp_model
        self.scenario = scenario
        self.case = case
        self.N_zones = N_zones
        self.coord_depot = coord_depot
        self.verbose = verbose
        
        self.run()
        return
    
    def run(self):
        path_save = current_path + '/results/'
        path_read = current_path + '/data/'

        if self.scenario == 'toy_case':
            # coords = [[0, 0], [-1, 2], [1, 0], [1, 8], [2, 0], [-2, 5]]
            # raw_data = [('1', -20.0), ('2', 20.0), ('3', -10.0), ('4', 17.0), ('5', -7.0)]
            # SystemGeneration(coord=coords,raw_data=raw_data)
            # n = len(coords)-1
            # zones_id = [i for i in range(n+1)]
            
            s = SystemGeneration(N_zones=self.N_zones,coord=None,raw_data=None)
            _, coords = s.generate_rawdata()
            s.generate_ODmatrix(od=1)
            zones_id = [i for i in range(self.N_zones+1)]
            print(zones_id)

        elif self.scenario == 'Louisville':
            d = DataProcess(N_zones=self.N_zones,coord_depot=self.coord_depot,path_save=path_save,path_read=path_read)
            zones_id = d.aggregate_od()
            # print(zones_id)
            coords = d.generate_coord(zones_id)
            # print(coords)

        else:
            print('Invalid scenario!')
            return

        distances = np.load(path_read + 'distance.npy')
        od_mat = np.load(path_read + 'od_mat.npy').astype(np.uint8)
        # print(od_mat)
        # print(distances)
        # print(zones_id)
        # print(f'shape od: {np.shape(od_mat)}')
        # print(f'shape distance: {np.shape(distances)}')

############ phase 1
        tfs = SystemSimulator(origin_destination_matrix=od_mat)
        # solution = 'simulation'
        solution = 'optimization'
        file_path = path_save + 'results - {}.csv'.format(solution)

        start_time1 = time()
        print('Solving relocation problem...')
        print()  
        o = Optimizer(parameters = self.paras_relocation, zones_id=zones_id,origin_destination_matrix=od_mat, distances=distances, optimization_horizon=self.optimization_horizon,
            look_ahead_horizon=self.look_ahead_horizon, system_simulator=tfs, verbose=self.verbose, file_path =file_path, mode = 'w')
        status1 = o.run_optimizer(solution)
        stop_time1 = time()
        elapsed_1 = stop_time1 - start_time1
        print(f'...results saved')
        print(f'{elapsed_1:.4f} seconds elapsed')
        print()

# ############# phase 2
        if status1 == 0 or status1 == 1:
            with open(path_save + 'relocation.json','r') as f:
                content = json.load(f)
            # print(content)
            raw_data = [content[t]['relo_operation'] for t in range(self.optimization_horizon)]
            print(f'raw_data: {raw_data}')
            print(f'coords: {coords}')

    # ######## plot zones
    #         # SystemGeneration().plot_zones(zones_id=zones_id,coord=coords,raw_data=raw_data)

            start_time2 = time()
            for t in range(self.optimization_horizon):
                print(f'Solving VRP over tf {t}...')
                print()  
                s = SolveVRP(self.vrp_model,zones_id,raw_data[t],coords,distances,self.paras_vrp,tf = t,path = path_save,case = self.case, verbose=self.verbose)
                stop_time2 = time()
            elapsed_2 = stop_time2 - start_time2
            print(f'...results saved')
            print(f'{elapsed_2:.4f} seconds elapsed') 
            print() 

            d = {
                'paras_relocation': [self.paras_relocation],
                'paras_vrp': [self.paras_vrp],
                'N_zones': self.N_zones,
                # 'elapsed_1': elapsed_1,
                'elapsed_2': elapsed_2           
            }
            # print(zones_data)
            # print(d)
            df = pd.DataFrame(d)
            df.to_csv(path_save + 'runtime.csv',mode='a')        
        

        # raw_data = [['219', -7.0], ['299', 7.0], ['219', -10.0], ['181', 10.0], ['219', -10.0], ['340', 10.0], ['307', -10.0], ['181', 10.0]]
#         s = SolveVRP(self.vrp_model,zones_id,raw_data,coords,distances,self.paras_vrp,tf = 9,path = path_save,case = 1, verbose=self.verbose)
        

        # overall_time_stop= time()
        # print(f'Overall time: {overall_time_stop-overall_time_start:.4f} seconds')
        return

if __name__ == '__main__':
    paras_relocation = {'N_scooter': 150, 'N_truck': 4, 'C_truck': 10, 'Incoming': 10, 'Cm': 0.00001, 'Cn': 0.01}
    optimization = 2
    lookahead = 1
        # C_truck - capacity of each truck
        # Incoming - incoming per trip
        # Cm - relocation cost / unit mileage cost
        # Cn - relocation cost / unit number of scooters cost
    paras_vrp = {'C_trucks': 200, 'N_trucks': 2, 'T': 700, 'Sr': 1, 'Tr': 0.00001}
    vrp_model = 's1'
        # T: duration of one timeframe
        # Sr: service time per scooter
        # Tr: travel time parameter (inverse of truck speed)
        # vrp_model: s1 - need large C&N_truck
        #             s2 - zones can be ignored
    
    # paras_relocation['N_scooter'] = 100
    # main(paras_relocation, optimization, lookahead, paras_vrp, vrp_model,'toy_case',N_zones=8)
    main(paras_relocation, optimization, lookahead, paras_vrp, vrp_model, 'Louisville', N_zones=7, coord_depot=[6,12])