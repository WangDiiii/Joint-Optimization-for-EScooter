from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit

import numpy as np
from plotnine import *
from time import time
import pandas as pd
import json
from SystemGeneration import *
import os

current_path = os.path.dirname(__file__)

TimeLimit = 1800*1000
TOLERANCE = 1e-5
ResultStatus = ['OPTIMAL','FEASIBLE','INFEASIBLE','UNBOUNDED','ABNORMAL','MODEL_INVALID','NOT_SOLVED']

class SolveVRP():
    def __init__(self, solution, zones_id, raw_data: list, coord, distance, paras: dict, tf, path: dict, case=0, verbose=True):
        self.solution = solution
        self.zones_id = zones_id
        self.raw_data = raw_data
        self.coord = coord
        self.zones_data = {}
        self.distance = distance
        self.N_trucks = paras['N_trucks']
        self.C_trucks = paras['C_trucks']
        self.T = paras['T']
        self.Sr = paras['Sr']
        self.Tr = paras['Tr']
        # self.Re = paras['Re']
        self.tf = tf
        self.path = path
        self.case = case
        self.verbose = verbose
        
        if self.raw_data: 
            self.result = {}
            self.run_optimizer()
        else:
            print('No information about zones!')
        return

    def run_optimizer(self):
        start = time()
        # print(f'Solving VRP over timeframe - {self.tf}')
        self.result['plotting'] = self.case
        self.result['time_frame'] = self.tf
        input_data = self.prepare_input()
        if self.solution == 's1': # Classical VRP
            route_lists = self.optimize_route_s1(input_data)
        elif self.solution == 's2': # Zones can be ignored
            route_lists = self.optimize_route_s2(input_data)
        else: 
            print('Choose 1 model to optimize!')
            return
        df_zones,plot = self.plot_zones(self.zones_data,routes = route_lists)
        
        stop = time()
        self.result['elapsed_time'] = '{:.4f}s'.format(stop-start)
        
        self.save_results()
        if self.verbose:
            print(self.result)
            print(plot)
        
        return
    
    def prepare_input(self):
        # for i in self.raw_data:
        #     print(type(i[1]))
        # self.result['raw_data'] = self.raw_data
        self.raw_data.insert(0,['0',0])
        z = []
        r = []
        nr = 0
        for i in self.raw_data:
            if i[0] not in z:
                z.append(i[0])
                nr = i[1]
                r.append(nr)
            else:
                r[z.index(i[0])] += i[1]       
        self.zones_data['zones'] = z
        self.zones_data['coordinates'] = [self.coord[self.zones_id.index(int(i))] for i in self.zones_data['zones']]
        self.zones_data['relocation'] = r
        # self.zones_data['relocation'].insert(0,0)
        self.zones_data['set'] = ['source' if r[z.index(i)]<0 else 'destination' for i in z[1:]]
        self.zones_data['set'].insert(0,'depot')

        source = []
        desti = []
        pick = []
        drop = []
        for i in range(len(self.zones_data['zones'])):
            n = self.zones_data['zones'][i]
            r = self.zones_data['relocation'][i]
            if r <= 0:
                source.append(int(n))
                pick.append(abs(int(r)))
            else:
                desti.append(int(n))
                drop.append(int(r))
        desti.append(0)
        drop.append(0)
        
        input_data = {}
        input_data['source'] = source
        input_data['destination'] = desti
        input_data['pick_up'] = pick
        input_data['drop_off'] = drop
        self.result['input_data'] = input_data
        # print(input_data)
        # print(self.zones_data)
        return input_data

    def optimize_route_s1(self,input_data):
        M = (self.C_trucks + 10)*2
        N = (self.T + 100)*2
        zones = input_data['source'] + input_data['destination'][:-1]
        r = input_data['pick_up'] + input_data['drop_off'][:-1]
        len_source = len(input_data['source'])
        len_zones=len(zones)
        dict_zones = {i:zones[i] for i in range(len_zones)}
        print(dict_zones)
        rd = [r[i] if i > len_source-1 else 0 for i in range(len(r))]
        # print(rd)

        # Create the linear solver with the CBC backend.
        self.solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

        # Create variables
        L = [[self.solver.IntVar(0, self.C_trucks,
            'L_' + str(i) + '_' + str(k)) if i!=0 else 0 for k in range(self.N_trucks)] for i in range(len_zones)]
        
        B = [[self.solver.NumVar(0, self.T,
            'B_' + str(i) + '_' + str(k)) if i!=0 else 0 for k in range(self.N_trucks)] for i in range(len_zones)]
        
        x = [[[self.solver.IntVar(0, 1,
            'x_' + str(i) + '_' + str(j) + '_' + str(k)) for k in range(self.N_trucks)] for j in range(len_zones)] for i in range(len_zones)]

        # 2. Add constraints for each resource

        for i in range(len_zones):
            for k in range(self.N_trucks):
                self.solver.Add(0 == x[i][i][k], 'not_in_zone' + str(i) + str(k))

        for i in range(1,len_zones):
            self.solver.Add(1 == sum(x[i][j][k] for k in range(self.N_trucks) for j in range(len_zones)), 'visit_once' + str(i))

        for k in range(self.N_trucks):
            self.solver.Add(1 >= sum(x[0][j][k] for j in range(len_zones)), 'departure_0_' + str(k))

        for k in range(self.N_trucks):
            self.solver.Add(1 >= sum(x[i][0][k] for i in range(len_zones)), 'return_0_'  + str(k))

        for k in range(self.N_trucks):
            for j in range(len_zones):
                self.solver.Add(sum(x[i][j][k] for i in range(len_zones)) == sum(x[j][i][k] for i in range(len_zones)),
                    'mass_conservation_'+ str(i) + '_'  + str(j) + '_' + str(k))

######## load constraint

        for i in range(len_source):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] >= r[i] - (1-x[i][j][k])*M,
                 'pick_up_bound_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(len_source,len_zones):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] <= self.C_trucks - r[i] + (1-x[i][j][k])*M,
                 'drop_off_bound_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(len_source):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] - (L[0][k] + r[i]) <= (1-x[0][i][k]) * M)
                self.solver.Add((L[0][k] + r[i]) - L[i][k] <= (1-x[0][i][k]) * M)
                self.solver.Add(x[i][0][k] == 0)

        for j in range(len_source,len_zones):
            for k in range(self.N_trucks):
                # self.solver.Add(L[j][k] <= (1-x[j][0][k]) * M)
                self.solver.Add(x[0][j][k] == 0)

        for k in range(self.N_trucks):
            self.solver.Add(sum(L[j][k] for j in range(1,len_zones)) <= sum(x[i][j][k] for i in range(len_zones) for j in range(1,len_zones)) * M,
            'load_constraint_' + str(k))

        for i in range(1,len_zones):
            for j in range(1,len_source):
                for k in range(self.N_trucks):
                    self.solver.Add(L[j][k] - (L[i][k] + r[j]) <= (1-x[i][j][k]) * M,
                     'pick_up_transition_left_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(1,len_source):
                for k in range(self.N_trucks):
                    self.solver.Add((L[i][k] + r[j]) - L[j][k] <= (1-x[i][j][k]) * M,
                     'pick_up_transition_right_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(len_source,len_zones):
                for k in range(self.N_trucks):
                    self.solver.Add(L[j][k] - (L[i][k] - r[j]) <= (1-x[i][j][k]) * M,
                     'drop_off_transition_left_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(len_source,len_zones):
                for k in range(self.N_trucks):
                    self.solver.Add((L[i][k] - r[j]) - L[j][k] <= (1-x[i][j][k]) * M,
                     'drop_off_transition_right_' + str(i) + '_' + str(j) + '_' + str(k))

        for k in range(self.N_trucks):
            for i in range(1,len_zones):
                for j in range(1,len_zones):
                    self.solver.Add(B[j][k] >= B[i][k] + self.Sr*r[i] + self.Tr*self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] - (1-x[i][j][k]) * N,
                     'time_' + str(i) + '_' + str(j) + '_' + str(k))
                    # self.solver.Add(B[j][k] >= B[i][k] + 3 - (1-x[i][j][k]) * M,
                    #  'time_' + str(i) + '_' + str(j) + '_' + str(k))
                    # self.solver.Add(B[-1][k] >= B[i][k],
                    #  'max_time_' + str(i) + '_' + str(j) + '_' + str(k))
        
        # 3. Maximize the objective function
        self.solver.Minimize(sum(x[i][j][k]*self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))))
        # self.solver.Minimize(sum(x[i][j][k] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))))
        self.solver.SetTimeLimit(TimeLimit)

        # Solve problem
        self.status = self.solver.Solve()
        print(f'{ResultStatus[self.status]} over time frame: {self.tf}')

        if self.status == pywraplp.Solver.OPTIMAL or self.status == 1:
            x_dict = {'x_' + str(dict_zones[i]) + '_' + str(dict_zones[j]) + '_' + str(k): x[i][j][k].solution_value() for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
            L_dict = {'L_' + str(dict_zones[i]) + '_' + str(k): L[i][k].solution_value() if i!=0 else L[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
            cost = {'c_' + str(dict_zones[i]) + '_' + str(dict_zones[j]) + '_' + str(k): x[i][j][k].solution_value()*self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
            route_lists = self.print_result(input_data,x_dict,L_dict,cost)

                # x_dict1 = {str(x[i][j][k]):x[i][j][k].solution_value() for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
                # L_dict1 = {str(L[i][k]): L[i][k].solution_value() if i!=0 else L[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
                # B_dict = {str(B[i][k]): B[i][k].solution_value() if i!=0 else B[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
                # print(f' - x1 : {x_dict1}')
                # xn1 = {k: v for k, v in x_dict1.items() if v!= 0}
                # print(f' - x_solver : {xn1}')
                # print(f' - L_solver : {L_dict1}')
                # print(f' - B : {B_dict}')
            return route_lists
        else:
            print(ResultStatus[self.status])
            # print(f'Result Status: {ResultStatus[status]} over time frame: {starting_time_frame}')
            return None

    def optimize_route_s2(self,input_data):
        M = (self.C_trucks + 10)*2
        N = (self.T + 100)*2
        zones = input_data['source'] + input_data['destination'][:-1]
        r = input_data['pick_up'] + input_data['drop_off'][:-1]
        len_source = len(input_data['source'])
        len_zones=len(zones)
        dict_zones = {i:zones[i] for i in range(len_zones)}
        print(dict_zones)
        rd = [r[i] if i > len_source-1 else 0 for i in range(len(r))]
        # print(rd)

        # Create the linear solver with the CBC backend.
        self.solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')

        # Create variables
        L = [[self.solver.IntVar(0, self.C_trucks,
            'L_' + str(i) + '_' + str(k)) if i!=0 else 0 for k in range(self.N_trucks)] for i in range(len_zones)]
        
        B = [[self.solver.NumVar(0, self.T,
            'B_' + str(i) + '_' + str(k)) if i!=0 else 0 for k in range(self.N_trucks)] for i in range(len_zones)]
        
        x = [[[self.solver.IntVar(0, 1,
            'x_' + str(i) + '_' + str(j) + '_' + str(k)) for k in range(self.N_trucks)] for j in range(len_zones)] for i in range(len_zones)]

        # 2. Add constraints for each resource

        for i in range(len_zones):
            for k in range(self.N_trucks):
                self.solver.Add(0 == x[i][i][k], 'not_in_zone' + str(i) + str(k))

        for i in range(1,len_zones):
            self.solver.Add(1 >= sum(x[i][j][k] for k in range(self.N_trucks) for j in range(len_zones)), 'visit_once' + str(i))

        for k in range(self.N_trucks):
            self.solver.Add(1 >= sum(x[0][j][k] for j in range(len_zones)), 'departure_0_' + str(k))

        for k in range(self.N_trucks):
            self.solver.Add(1 >= sum(x[i][0][k] for i in range(len_zones)), 'return_0_'  + str(k))

        for k in range(self.N_trucks):
            for j in range(len_zones):
                self.solver.Add(sum(x[i][j][k] for i in range(len_zones)) == sum(x[j][i][k] for i in range(len_zones)),
                    'mass_conservation_'+ str(i) + '_'  + str(j) + '_' + str(k))

##### load constraint

        for i in range(len_source):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] >= r[i] - (1-x[i][j][k])*M,
                 'pick_up_bound_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(len_source,len_zones):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] <= self.C_trucks - r[i] + (1-x[i][j][k])*M,
                 'drop_off_bound_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(len_source):
            for k in range(self.N_trucks):
                self.solver.Add(L[i][k] - (L[0][k] + r[i]) <= (1-x[0][i][k]) * M)
                self.solver.Add((L[0][k] + r[i]) - L[i][k] <= (1-x[0][i][k]) * M)
                self.solver.Add(x[i][0][k] == 0)

        for j in range(len_source,len_zones):
            for k in range(self.N_trucks):
                # self.solver.Add(L[j][k] <= (1-x[j][0][k]) * M)
                self.solver.Add(x[0][j][k] == 0)

        for i in range(1,len_zones):
            for j in range(1,len_source):
                for k in range(self.N_trucks):
                    self.solver.Add(L[j][k] - (L[i][k] + r[j]) <= (1-x[i][j][k]) * M,
                     'pick_up_transition_left_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(1,len_source):
                for k in range(self.N_trucks):
                    self.solver.Add((L[i][k] + r[j]) - L[j][k] <= (1-x[i][j][k]) * M,
                     'pick_up_transition_right_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(len_source,len_zones):
                for k in range(self.N_trucks):
                    self.solver.Add(L[j][k] - (L[i][k] - r[j]) <= (1-x[i][j][k]) * M,
                     'drop_off_transition_left_' + str(i) + '_' + str(j) + '_' + str(k))

        for i in range(1,len_zones):
            for j in range(len_source,len_zones):
                for k in range(self.N_trucks):
                    self.solver.Add((L[i][k] - r[j]) - L[j][k] <= (1-x[i][j][k]) * M,
                     'drop_off_transition_right_' + str(i) + '_' + str(j) + '_' + str(k))

        for k in range(self.N_trucks):
            for i in range(1,len_zones):
                for j in range(1,len_zones):
                    self.solver.Add(B[j][k] >= B[i][k] + self.Sr*r[i] + self.Tr*self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] - (1-x[i][j][k]) * N,
                     'time_' + str(i) + '_' + str(j) + '_' + str(k))
                    # self.solver.Add(B[j][k] >= B[i][k] + 3 - (1-x[i][j][k]) * M,
                    #  'time_' + str(i) + '_' + str(j) + '_' + str(k))
                    # self.solver.Add(B[-1][k] >= B[i][k],
                    #  'max_time_' + str(i) + '_' + str(j) + '_' + str(k))
        
        # 3. Maximize the objective function
        # self.solver.Minimize(sum(x[i][j][k]*self.distance[dict_zones[i]][dict_zones[j]] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))))
        self.solver.Maximize(sum(x[i][j][k]*rd[j]*10000 - x[i][j][k] * self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))))
        # self.solver.Minimize(sum(x[i][j][k] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))))
        self.solver.SetTimeLimit(TimeLimit)

        # Solve problem
        self.status = self.solver.Solve()

        if self.status == pywraplp.Solver.OPTIMAL or self.status == 1:
            x_dict = {'x_' + str(dict_zones[i]) + '_' + str(dict_zones[j]) + '_' + str(k): x[i][j][k].solution_value() for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
            L_dict = {'L_' + str(dict_zones[i]) + '_' + str(k): L[i][k].solution_value() if i!=0 else L[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
            cost = {'c_' + str(dict_zones[i]) + '_' + str(dict_zones[j]) + '_' + str(k): x[i][j][k].solution_value()*self.distance[self.zones_id.index(dict_zones[i])][self.zones_id.index(dict_zones[j])] for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
            route_lists = self.print_result(input_data,x_dict,L_dict,cost)

                # x_dict1 = {str(x[i][j][k]):x[i][j][k].solution_value() for k in range(self.N_trucks) for i in range(len(zones)) for j in range(len(zones))}
                # L_dict1 = {str(L[i][k]): L[i][k].solution_value() if i!=0 else L[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
                # B_dict = {str(B[i][k]): B[i][k].solution_value() if i!=0 else B[0][k] for k in range(self.N_trucks) for i in range(len(zones))}
                # print(f' - x1 : {x_dict1}')
                # xn1 = {k: v for k, v in x_dict1.items() if v!= 0}
                # print(f' - x_solver : {xn1}')
                # print(f' - L_solver : {L_dict1}')
                # print(f' - B : {B_dict}')
            return route_lists
        else:
            print(ResultStatus[self.status])
            # print(f'Result Status: {ResultStatus[status]} over time frame: {starting_time_frame}')
            return None

    def plot_zones(self,zones_data,routes = None):
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
            'labels': labels,
            'Y': y,
            'X': x,
            'set': s
        }
        # print(zones_data)
        # print(d)
        df_zones = pd.DataFrame(d)
        df_zones.to_csv(self.path + 'zones.csv',mode='w')
        # print(df_zones)

        p = (ggplot(df_zones,aes(x='X',y='Y', group='set', color='set')))
        p = (p + theme_bw()
            + geom_point(size=3)
            + geom_text(aes(x='X',y='Y',label='labels'),nudge_y=0.5)
            + xlim(min(x)-1,max(x)+1)
            + ylim(min(y)-1,max(y)+1)
            + scale_color_discrete(guide=False)
            # + labs(x=labelx, y=labely, title=title)
            # + theme(figure_size=(8, 6), aspect_ratio=4/9)
            # + theme(legend_position=(0.95, 0.55), legend_direction="vertical")
            )
        # print(p)
        if routes is None or len(routes)==0:
            ggplot.save(p, filename=self.path + 'plot_{}_{}.pdf'.format(self.case,self.tf), verbose = False) 
        else:
            color = []
            for i in range(len(routes)):
                color.append('#%06X' % np.random.randint(0, 0xFFFFFF))
            c = 0
            for route in routes:
                coords = []
                for i in route:
                    ind = zones_data['zones'].index(i)
                    coord = zones_data['coordinates'][ind]
                    coords.append(coord)
                x0 = [i[0] for i in coords[:-1]]
                y0 = [i[1] for i in coords[:-1]]
                xe = [i[0] for i in coords[1:]]
                ye = [i[1] for i in coords[1:]]

                for i in range(len(x0)):
                    p += geom_segment(aes(x = x0[i], y = y0[i], xend = xe[i], yend = ye[i]),
                    arrow = arrow(), size = 0.5, color=color[c])
                ggplot.save(p, filename = self.path + 'plot_{}_{}.pdf'.format(self.case,self.tf), verbose = False) 
                c += 1
        return df_zones, p

    def print_result(self,input_data,x_dict,L_dict,cost):
        L = [{k:v for k,v in L_dict.items() if k.split('_')[-1] == str(i)} for i in range(self.N_trucks)]
        cn = {k: v for k, v in cost.items() if v!= 0}
        sum_cn = sum(cn.values())
        xn = {k: v for k, v in x_dict.items() if v!= 0}
        x = [{k:v for k,v in xn.items() if k.split('_')[-1] == str(i)} for i in range(self.N_trucks)]
        routes = [[str(0)] for _ in range(len(x))]
        route_lists = [[str(0)] for _ in range(len(x))]
        print('while...')
        for ktruck in range(len(x)):
            i = str(0)
            keys = list(x[ktruck].keys())
            
            while keys:
                for key in keys:
                    k = key.split('_')
                    if k[1] == i:
                        if int(k[2]) in input_data['source']:
                            n = (str(-1*input_data['pick_up'][input_data['source'].index(int(k[2]))]))
                        else:
                            n = (str(input_data['drop_off'][input_data['destination'].index(int(k[2]))]))
                        s = '-> {}({})'.format(k[2],n)
                        route_lists[ktruck].append(k[2])
                        routes[ktruck].append(s)
                        i = k[2]
                        keys.remove(key)
        for i in range(len(routes)):
            routes[i] = " ".join(routes[i])

        self.result['route_cost'] = sum_cn
        self.result['optimal_route'] = routes
        self.result['truck_load'] = L
        self.result['N_route'] = len(cn)
        self.result['cost_each_route'] = cn

            # print(f'The minimize route cost = {sum_cn}')
            # print(f'route cost: {cn}')
            # # print(f' - x : {x_dict}')
            # print(f' - x : {x}')
            # print(f' - optimal route: {routes}')
            # print(f' - L : {L}')
            # # print(route_list)

        if self.verbose:
            print()
            print(f'The model solved in {self.solver.wall_time()*0.001:.6f} seconds in {self.solver.iterations()} iterations')
            # print(f'Optimal value of objective = {self.solver.Objective().Value()}')
        
        return route_lists

    def save_results(self):
        if self.status == 0 or self.status == 1:
####### save to json
            try:
                with open(self.path + 'result.json','r') as f:
                    content = json.load(f)
            except json.decoder.JSONDecodeError:
                content = []
            except FileNotFoundError:
                content = []
                with open(self.path + 'result.json','w') as f:                
                    json.dump(content,f)
            # print(content)
            content.append(self.result)
            with open(self.path + 'result.json','w') as f2:
                json.dump(content,f2)
        
####### save to csv
        if self.status == 0 or self.status == 1:
            columns=['plotting','time_frame','route_cost','optimal_route','N_route','elapsed_time']
            results = pd.DataFrame(columns=columns)
            if self.result['plotting'] == self.case:
                data = [self.result[k] for k in self.result.keys() if k in columns]
        # # results = results.append(data,ignore_index=True)
            results.loc[len(results.index)] = data
        else:
            columns=['plotting','time_frame','N_route','elapsed_time']
            results = pd.DataFrame(columns=columns)
            if self.result['plotting'] == self.case:
                data = [self.result[k] for k in self.result.keys() if k in columns]
        # # results = results.append(data,ignore_index=True)
            data.insert(-2,0)
            results.loc[len(results.index)] = data     
        results.to_csv(self.path + 'result.csv',mode='a')
        # mode = 'a' / 'w'
        if self.verbose:
            print(f'The results are saved in {self.path}')
        
        return

if __name__ == '__main__':
    tf = 0
    case = 0
    path_current = current_path + '/'
    path_save = current_path + '/resultsVRP/'
    # paras = {'C_trucks': 20, 'N_trucks': 2, 'T': 70, 'Sr': 1, 'Tr': 1}
# T: duration of one timeframe
# Sr: service time per scooter
# Tr: travel time parameter (inverse of truck speed)

####### random zones
    # paras = {'C_trucks': 20, 'N_trucks': 2, 'T': 70, 'Sr': 1, 'Tr': 1}
    # raw_data, coords = SystemGeneration(N_zones=4,coord=None,raw_data=None).generate_rawdata()

####### scenario 0
    # paras = {'C_trucks': 15, 'N_trucks': 2, 'T': 40, 'Sr': 1, 'Tr': 1}
    # raw_data = [('0', 0), ('1', -4), ('2', 4), ('3', -7), ('4', 7)]
    # coords = [[0, 0], [-12, 3], [-14, 0], [10, -5], [11, 2]]
    # SystemGeneration(coord=coords,raw_data=raw_data)

# ####### scenario 1
#     paras = {'C_trucks': 20, 'N_trucks': 2, 'T': 70, 'Sr': 1, 'Tr': 1}
#     coords = [[0, 0], [-6, -2], [-11, 5], [-15, -4], [-6, 10], [6, 7], [9, -4], [13, -3], [9, -3]]
#     raw_data = [('1', -10.0), ('3', 10.0), ('2', -10.0), ('3', 10.0), ('6', -9.0), ('8', 9.0), ('7', -8.0), ('8', 8.0)]
#     SystemGeneration(coord=coords,raw_data=raw_data)

# ###### run
#     distances = np.load(path_current + 'distances.npy')
#     s = SolveVRP(raw_data,coords,distances,paras,tf = tf,path = path_save,case = 1,verbose = True)

# ####### plotting
#     paras = {'C_trucks': 20, 'N_trucks': 20, 'T': 700, 'Sr': 1, 'Tr': 1}
#     raw_data, coords = SystemGeneration(N_zones=4,coord=None,raw_data=None).generate_rawdata()
#     distances = np.load(path_current + 'distances.npy')
#     print(raw_data)

#     range_Ct = [20]
#     range_Nt = np.arange(1,2).tolist()
#     range_Time = [100]
#     elap = []
#     N_route = []
#     Time = []
#     Ct = []
#     Nt = []
#     for j in range_Time:
#         # for i in Time:
#         for i in range_Nt:
#             paras['N_trucks'] = i
#             # paras['C_trucks'] = i
#             paras['T'] = j
#             Nt.append(i)
#             Time.append(j)
#             c = 'T{}_N{}'.format(paras['T'],paras['N_trucks'])
#             # c = 'C{}_T{}'.format(paras['C_trucks'],paras['T'])  
#             start = time()
#             SolveVRP(raw_data,coords,distances,paras,tf = tf,path = path_save,case = c,verbose = True)
#             end = time()
#             elap.append(end-start)
#             with open(current_path + '/resultsVRP/'+'result.csv') as f:
#                 last_line = f.readlines()[-1]
#             nr = last_line.split(',')[-2]
#             N_route.append(nr)
#             d = {
#                 'elap': elap
#             }
#             df = pd.DataFrame(d)
#             df.to_csv(current_path + '/resultsVRP/df-runtime.csv'.format(c),mode='w')
#     d = {
#         'Time': Time,
#         'Nt': Nt,
#         'N_route': N_route,
#         'elap': elap
#     }
#     df = pd.DataFrame(d)
#     df.to_csv(current_path + '/resultsVRP/df-runtime.csv'.format(c),mode='w')    
#     print(df)

        