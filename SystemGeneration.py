import pandas as pd
from plotnine import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from time import time
import math
import os

current_path = os.path.dirname(__file__)
np.random.seed(6)


class SystemGeneration():
    def __init__(self, N_zones=None, coord=None, raw_data=None, N_timeframes=30):
    
        self.N_zones = N_zones
        self.N_timeframes = N_timeframes
        self.coord = coord
        self.raw_data = raw_data

        if self.coord is not None and self.raw_data is not None:
            self.generate_distances()
            n=[]
            for i in self.raw_data:
                if i[0] not in n:
                    n.append(i[0])
            # print(n)
            # self.coord = [self.coord[int(i)] for i in n]
            self.N_zones = len(n)
        elif self.N_zones is not None:
            if self.N_zones%2 == 0:
                self.cut = self.N_zones/2
            else:
                self.cut = (self.N_zones+1)/2
            self.coord = self.generate_coord()
            self.generate_distances()
            self.raw_data = self.generate_rawdata()
        return

    def generate_coord(self):
        coord = [[0,0]]
        for i in range(self.N_zones):
            if i < self.cut:
                coord.append([np.random.randint(-15,-5), np.random.randint(-5,15)])
            else:
                coord.append([np.random.randint(5,15), np.random.randint(-5,15)])
        # print(coord)
        return coord

    def generate_distances(self):
        A=np.array(self.coord)
        distA=pdist(A,metric='euclidean')
        distances = squareform(distA)
        # print(distances)
        np.save(current_path + '/data/' + 'distance',distances)

    def generate_rawdata(self):
        raw_data = []
        z = list(range(1,self.N_zones+1))
        r = np.random.randint(-10,10,size=self.N_zones-1)
        r = [i if i!=0 else np.random.randint(1,10) for i in r]
        dif = sum(r)
        if dif == 0:
            r[-1] += np.random.randint(1,10)
            dif = sum(r)
        r.append(-dif)
        np.random.shuffle(r)

        i = 0
        while r:
            raw_data.append((str(z[i]), r.pop(0)))
            i+=1
        
        return raw_data, self.coord

    def plot_zones(self,zones_id,coord,raw_data):
        raw_data.insert(0,('0',0))
        z = []
        r = []
        nr = 0
        for i in raw_data:
            if i[0] not in z:
                z.append(i[0])
                nr = i[1]
                r.append(nr)
            else:
                r[z.index(i[0])] += i[1]
        zones_data = {}
        zones_data['zones'] = z
        zones_data['coordinates'] = [coord[zones_id.index(int(i))] for i in z]
        zones_data['relocation'] = r
        # zones_data['relocation'].insert(0,0)
        zones_data['set'] = ['source' if r[z.index(i)]<0 else 'destination' for i in z[1:]]
        zones_data['set'].insert(0,'depot')

        labels = []
        x = []
        y = []
        s = []
        for i in range(len(zones_data['zones'])):
            ln = zones_data['zones'][i]
            lr = zones_data['relocation'][i]
            labels.append([ln,lr])
            x.append(zones_data['coordinates'][i][0])
            y.append(zones_data['coordinates'][i][1])
            s.append(zones_data['set'][i])
        d = {
            'Y': y,
            'X': x,
            'set': s,
            'labels': labels
        }
        df = pd.DataFrame(d)
        # print(df)

        p = (ggplot(df,aes(x='X',y='Y', group='set', color='set')))
        p = (p + theme_bw()
            + geom_point()
            + geom_text(aes(x='X',y='Y',label='labels'),nudge_y=0.5)
            + xlim(min(x)-1,max(x)+1)
            + ylim(min(y)-1,max(y)+1)
            # + labs(x=labelx, y=labely, title=title)
            # + theme(figure_size=(8, 6), aspect_ratio=4/9)
            # + theme(legend_position=(0.95, 0.55), legend_direction="vertical")
            )
        print(p)

    def generate_ODmatrix(self,od=0):
        N_zones = self.N_zones+1
# All goes to zone 2
        if od == 0:
            od_mat = np.zeros((self.N_timeframes,N_zones,N_zones),dtype= int)
            for t in range(np.shape(od_mat)[0]):
                for i in range(1,np.shape(od_mat)[1]):
                    for j in range(1,np.shape(od_mat)[2]):
                        # if (i== 0) or (j == 0):
                        #     od_mat[t][i][j] = 0
                        if (i!= 3) & (j == 3):
                            od_mat[t][i][j] = 50

# random demand same for all timeframe        
        if od == 1:
            od_mat = np.random.randint(0,10,size=[N_zones,N_zones])
            for i in range(np.shape(od_mat)[1]):
                # for j in range(np.shape(od_mat)[2]):
                #     if (i== 0) or (j == 0):
                #         od_mat[t][i][j] = 0
                od_mat[0][i] = 0
                od_mat[i][0] = 0
                od_mat[i][i] = 0
            od_mat = np.tile(od_mat,(self.N_timeframes,1))
            od_mat = od_mat.reshape((self.N_timeframes,N_zones,N_zones)) 

        if od == 2:
            od = np.zeros((1,N_zones,N_zones),dtype= int)
            od[0][1][2] = 1
            od[0][2][2] = 2
            od[0][3][2] = 3
            od_mat = np.concatenate((od,od),axis=0)
            for i in range(8):
                od_mat = np.concatenate((od,od_mat),axis=0)



        # print(od_mat[0])
        # print(np.shape(od_mat))
        # print(sum(sum(od_mat[0])))
        np.save(current_path + '/data/' + 'od_mat',od_mat)    

if __name__ == '__main__':
    N_zones = 4
    N_timeframes = 30
    
    # raw_data, coords = SystemGeneration(N_zones=4,coord=None,raw_data=None).generate_rawdata()

    coords = [[0, 0], [-6, -2], [-11, 5], [-15, -4], [-6, 10], [6, 7], [9, -4], [13, -3], [9, -3]]
    raw_data = [('1', -10.0), ('3', 10.0), ('2', -10.0), ('3', 10.0), ('6', -9.0), ('8', 9.0), ('7', -8.0), ('8', 8.0)]
    SystemGeneration(coord=coords,raw_data=raw_data)


    # coords = [[0, 0], [-6, -2], [-11, 5], [-15, -4], [-6, 10], [6, 7], [9, -4], [13, -3], [9, -3]]
    # raw_data = [('1', -10.0), ('3', 10.0), ('2', -10.0), ('3', 10.0), ('6', -9.0), ('8', 9.0), ('7', -8.0), ('8', 8.0)]
    # # coords = [[0, 0], [-6, -2], [-11, 5], [-15, -4]]
    # # raw_data = [('1', -10.0), ('3', 10.0), ('2', -10.0), ('3', 10.0)]


    # zdata = s.generate_zonesdata()
    # # print(rdata)
    # # print(zdata)
    # s.plot_zones(zdata)