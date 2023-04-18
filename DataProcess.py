import numpy as np
from plotnine import *
from time import time
import pandas as pd
import json
from scipy.spatial.distance import cdist
from SystemGeneration import *
from Optimize_relocation import *
import os

current_path = os.path.dirname(__file__)

TimeLimit = 3600*1000
TOLERANCE = 1e-5

class DataProcess():
    def __init__(self, N_zones,coord_depot,path_save,path_read):
        self.N_zones = N_zones
        self.coord_depot = coord_depot
        self.path_save = path_save
        self.path_read = path_read

        self.sorted_rows,self.sorted_sums = self.filter_zones()
        # selected_zones = self.aggregate_od()
        # coords = self.generate_coord(selected_zones)
    
    def filter_zones(self):
######### total od and sort
        od_mat = np.zeros((378,378),dtype = int)
        count_by_hour = pd.read_csv(self.path_read + 'new_od_total_count_by_hour.csv')
        # hours = [[0,1,2,3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23]]
        hours = [i for i in range(24)]
        # print(hours)

        for i in hours:
            # file_path = 'main/data/od_total_count_wide_{}.csv'.format(i)
            # c = pd.read_csv(file_path)
            # count = c.iloc[:3,:10]
            count = count_by_hour[(count_by_hour['hour'] == i)]
            # count = c.iloc[:100]
            # print(count)       
            for row in count.itertuples():
                o = int(row[2])     
                d = int(row[3])
                od_mat[o][d] += row[5]
        # print(od_mat)
        # print(np.shape(od_mat))
        # print(type(od_mat))

        df = pd.DataFrame(od_mat)
        # print(df)
        df.to_csv(self.path_read + 'total_od.csv')

        sums = []
        for i in range(len(df.index)):
            row_sum = df.iloc[i, :].sum()
            col_sum = df.iloc[:, i].sum()
            sums.append(row_sum + col_sum)
        #     print(row_sum)
        #     print(col_sum)
        sorted_rows = sorted(range(len(sums)), key=lambda k: sums[k], reverse = True)
        sorted_sums = sorted(sums, reverse = True)
        # print(sorted_rows)
        # print(sorted_sums)
        # print(sums)
        return sorted_rows,sorted_sums

    def aggregate_od(self):
#########select zones
        N_zones = self.N_zones
        selected_sums = self.sorted_sums[:N_zones]
        selected_zones = self.sorted_rows[:N_zones]
        selected_zones.insert(0,0)
        print(f'{N_zones} zones have been selected, covered {sum(selected_sums)/sum(self.sorted_sums):.2%} trips')
        # print(selected_sums)
######### aggragate od_mat
        N_timeframes = 7
        od_mat = np.zeros((N_timeframes,1476,1476),dtype = int)
        count_by_hour = pd.read_csv(self.path_read + 'new_od_total_count_by_hour.csv')
        hours = [[0,1,2,3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23]]

        for j in hours:
            ind = hours.index(j)
            for i in hours[ind]:
                count = count_by_hour[(count_by_hour['hour'] == i)]      
                for row in count.itertuples():
                    o = int(row[2])     
                    d = int(row[3])
                    od_mat[ind][o][d] += row[5]

        result = od_mat[:, selected_zones][:, :, selected_zones]
        # print(result)
        # print(np.shape(result))
        np.save(self.path_read + 'od_mat',result)
        return selected_zones 

    def generate_coord(self,selected_zones):
######### coords and corresponding distances
        grid = pd.read_csv(self.path_read + 'new_grid_matrix.csv')
        grid = grid.drop(grid.columns[[0]],axis = 1)
        # print(grid)
        coords = [self.coord_depot]
        for value in selected_zones[1:]:
            # print(value)
            rows, cols = (grid == value).to_numpy().nonzero()
            coord = [cols[0], rows[0]]
            coords.append(coord)
        # print(coords)
        distances_depot = cdist(np.array(coords), np.array([self.coord_depot])).ravel().tolist()
        # s = pd.Series(distances_depot,index=selected_zones)
        # s = s*200
        # print(s)

        distances_depot = [i*200 for i in distances_depot]
        # print(distances_depot)

        distance = pd.read_csv(self.path_read + 'new_distance_matrix.csv')
        distance = distance.drop(distance.columns[[0]],axis = 1)

        # subset the data frame with the selected rows and columns
        df_subset = distance.iloc[selected_zones[1:],selected_zones[1:]]
        # df_subset = distance.iloc[selected_zones,selected_zones]

        df_subset = df_subset.values
        df_subset = np.insert(df_subset, 0, distances_depot[1:], axis=1)
        df_subset= np.insert(df_subset, 0, distances_depot, axis=0)


        # print(df_subset)
        # print(np.shape(df_subset))
        # print(type(df_subset))
        np.save(self.path_read + 'distance',df_subset)
        return coords

if __name__ == '__main__':
    DataProcess(N_zones=5,coord_depot=[0,0])

    paras = {'N_scooter': 50, 'N_truck': 4, 'C_truck': 10, 'Incoming': 10, 'Cm': 0.00001, 'Cn': 0.01}
# C_truck - capacity of each truck
# Incoming - incoming per trip
# Cm - relocation cost / unit mileage cost
# Cn - relocation cost / unit number of scooters cost

    # SystemGeneration(4,10).generate_ODmatr
    distances = np.load(current_path + '/distance.npy')
    od_mat = np.load(current_path + '/od_mat.npy').astype(np.uint8)
    # print(od_mat)
    # print(distances)
    print(np.shape(od_mat))
    print(np.shape(distances))
    tfs = SystemSimulator(origin_destination_matrix=od_mat)

    # solution = 'simulation'
    solution = 'optimization'
    file_path = current_path + '/results - {}.csv'.format(solution)
    o = Optimizer(parameters = paras, origin_destination_matrix=od_mat, distances=distances, optimization_horizon=1,
        look_ahead_horizon=1, system_simulator=tfs, verbose=True, file_path =file_path, mode = 'a')
    o.run_optimizer(solution)