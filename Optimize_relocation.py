from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit

import numpy as np
import random
from time import time
import pandas as pd
import json
from SystemSimulator import SystemSimulator
from SystemGeneration import *
import os

current_path = os.path.dirname(__file__)

random.seed(60)
TimeLimit = 1800*1*1000
ResultStatus = ['OPTIMAL','FEASIBLE','INFEASIBLE','UNBOUNDED','ABNORMAL','MODEL_INVALID','NOT_SOLVED']

TOLERANCE = 1e-5

class Optimizer():
    def __init__(self,
                 parameters: dict,
                 zones_id: list,
                 origin_destination_matrix: np.array,
                 distances: np.array,
                 optimization_horizon: int,
                 look_ahead_horizon: int,
                 system_simulator: SystemSimulator,
                 file_path: str,
                 mode: str,
                 verbose: bool = True):
        
        self.parameters = parameters
        vehicle_number = self.parameters['N_scooter']
        self.N_truck = self.parameters['N_truck']
        self.C_truck = self.parameters['C_truck']
        maximum_relocation = self.C_truck * self.N_truck
        self.I = self.parameters['Incoming']
        self.Cm = self.parameters['Cm']
        self.Cn = self.parameters['Cn']
        self.parameters['t_opti'] = optimization_horizon
        self.parameters['t_look'] = look_ahead_horizon
        self.zones_id = zones_id
        self.mode = mode

        if vehicle_number < 0:
            raise ValueError('The number of vehicles can\'t be negative')
        self.__vehicle_number = vehicle_number
        if maximum_relocation < 0:
            raise ValueError('The maximum number of vehicles to be relocated can\'t be negative')
        self.__origin_destination_matrix = origin_destination_matrix
        self.__maximum_relocation = maximum_relocation
        self.__time_frames_number, self.__zones_number, _ = self.__origin_destination_matrix.shape

        self.__origin_destination_probability = np.zeros(
            (self.__time_frames_number, self.__zones_number, self.__zones_number), dtype=float)
        for t in range(self.__time_frames_number):
            row_sums = self.__origin_destination_matrix[t].sum(axis=1)
            self.__origin_destination_probability[t] = self.__origin_destination_matrix[t] / row_sums[:, np.newaxis]
        self.__origin_destination_probability = np.nan_to_num(self.__origin_destination_probability)
        # print(f'probability:{self.__origin_destination_probability}')

        # computing incoming and outgoing demand arrays
        self.__outgoing_demand = np.zeros((self.__time_frames_number, self.__zones_number), dtype=np.uint8)
        self.__incoming_demand = np.zeros((self.__time_frames_number, self.__zones_number), dtype=np.uint8)
        for t in range(self.__time_frames_number):
            self.__outgoing_demand[t] = self.__origin_destination_matrix[t].sum(axis=1)
            self.__incoming_demand[t] = self.__origin_destination_matrix[t].sum(axis=0)
        # print(f'out demand: {self.__outgoing_demand}')
        # print(self.__outgoing_demand.shape)

        if optimization_horizon < 1:
            raise ValueError('Unvalid number of optimization horizon')

        if look_ahead_horizon < 1:
            raise ValueError('Unvalid number of look ahead horizon')

        if self.__time_frames_number < optimization_horizon + look_ahead_horizon -1:
            raise ValueError('Not enough time frame data are provided to optimize over the optimization horizon')

        self.__look_ahead_horizon = look_ahead_horizon
        self.__optimization_horizon = optimization_horizon

        self.__possible_origins = [
            [[i for i in range(self.__zones_number) if self.__origin_destination_matrix[t, i, j] > 0] for j in
             range(self.__zones_number)] for t in range(self.__optimization_horizon + self.__look_ahead_horizon -1)]
        self.__possible_destinations = [
            [[j for j in range(self.__zones_number) if self.__origin_destination_matrix[t, i, j] > 0] for i in
             range(self.__zones_number)] for t in range(self.__optimization_horizon + self.__look_ahead_horizon -1)]
        self.__possible_destinations_probabilities = [[[self.__origin_destination_probability[t, i, j] for j in
                                                        self.__possible_destinations[t][i]] for i in
                                                       range(self.__zones_number)] for t in
                                                      range(self.__optimization_horizon + self.__look_ahead_horizon -1)]

        self.__start_zones = [[]] * (self.__optimization_horizon + self.__look_ahead_horizon -1)
        for time_frame in range(self.__optimization_horizon + self.__look_ahead_horizon -1):
            for i in range(self.__zones_number):
                self.__start_zones[time_frame].extend([i] * self.__outgoing_demand[time_frame][i])
        # print(f'start zones: {self.__start_zones}')

        self.__trips = [[[]] * self.__zones_number] * (self.__optimization_horizon + self.__look_ahead_horizon -1)
        for time_frame in range(self.__optimization_horizon + self.__look_ahead_horizon -1):
            for i in range(self.__zones_number):
                for j in range(self.__zones_number):
                    if self.__origin_destination_matrix[time_frame][i, j] > 0:
                        self.__trips[time_frame][i].extend([j] * self.__origin_destination_matrix[time_frame][i, j])

        self.distances = distances
        self.system_simulator = system_simulator

        if not isinstance(verbose, bool):
            raise ValueError('Verbose can be either False (deactivated) or True (activated)')

        self.__verbose = verbose
        self.path = file_path

    def run_optimizer(self,solution):
        s0 = [0] * self.__zones_number
        i = 1
        for _ in range(self.__vehicle_number):
            s0[i] += 1

            if i == self.__zones_number - 1:
                i = 1
            else:
                i += 1
        current_system_state = s0
        # current_system_state = [0,15,25,60]
        # current_system_state = [60,15,25,0]

        title = False
        EOsum = 0
        total_revenue = 0
        current_system_state0 = current_system_state

        # if solution == 'simulation':
        #     for t in range(self.__optimization_horizon):
        #         c = current_system_state
        #         start_time = time()
        #         s = True
        #         # state, r, x, w, opt_EI, opt_EO, r_dict, w_dict, x_dict, status = self.__optimize_relocation(starting_time_frame=t,
        #         #                                             initial_state=current_system_state) 
        #         try:
        #             state, r, x, w, opt_EI, opt_EO, r_dict, w_dict, x_dict, status = self.__optimize_relocation(starting_time_frame=t,
        #                                     initial_state=current_system_state)
        #         except Exception:
        #             s = False
        #             print(f'Over time-frame {t} with lookahead horizon {self.__look_ahead_horizon}')
        #             print('[-1] The solver could not find an optimal solution')
        #             print()
        #             break

        #         # opt_state,opt_revenue, opt_EO = self.__optimize_objective(r=r,x=x,w=w,initial_state=current_system_state,starting_time_frame=t) 
        #         current_system_state, states, satisfied_demand, actual_revenue = self.__simulate_revenue(s=current_system_state, r=r, time_frame=t,x_dict=x_dict)
        #         current_system_state0, states0, satisfied_demand0, actual_revenue0 = self.__simulate_revenue(s=current_system_state0, r=[[0]*self.__zones_number], time_frame=t,x_dict = dict())

        #         if s:
        #             self.columns=[' ','Parameters','timeframe #','Initial system state','Current system state','Relocation operation','Relocation number',
        #                     'Relocation Path','Relocation Cost','Actu_EO','Actu_Revenue','Zero_EO','Zero_Revenue','OD matrix','Distance matrix']
        #             if not title:
        #                 results = pd.DataFrame(columns=self.columns)
        #                 # results.to_csv(current_path + '/results.csv', mode='a')
        #                 title = True
        #             data, relo_operation = self.__save_results(time_frame=t, opt_revenue = actual_revenue, opt_EO = satisfied_demand, initial_system_state=c, 
        #                                         current_system_state=current_system_state, r_dict=r_dict, w_dict=w_dict, x_dict=x_dict)
        #             data.insert(-2,[sum(satisfied_demand0),satisfied_demand0])
        #             data.insert(-2, actual_revenue0)
        #             # results = results.append(data,ignore_index=True)
        #             results.loc[len(results.index)] = data

        #     stop_time = time()
        #     if s:
        #         # print(f'Satisfied demand over time-frame {t} is {satisfied_demand[0]} out of {sum(sum(self.__origin_destination_matrix[t]))}')
        #         print(f'{stop_time - start_time:.4f} seconds elapsed')
        #         print()
        #         results.to_csv(self.path,mode=self.mode)
        #         # mode = 'a' / 'w'
        #         with open(current_path + '/relocation.json','w') as f:
        #             json.dump(relo_operation,f)
        #         print(f'The results are saved in {self.path}')
        #     return

        if solution == 'optimization':
            opti_r = []
            for t in range(self.__optimization_horizon):
                c = current_system_state
                start_time = time()
                s = True
                # state, r, x, w, opt_EI, opt_EO, r_dict, w_dict, x_dict, status = self.__optimize_relocation(starting_time_frame=t,
                #                                             initial_state=current_system_state) 
                try:
                    state, r, x, w, opt_EI, opt_EO, r_dict, w_dict, x_dict = self.__optimize_relocation(starting_time_frame=t,
                                            initial_state=current_system_state)
                except Exception:
                    s = False
                    break
                # print(r)
                state = state[0]
                opt_EO = opt_EO[0]
                opt_EI = opt_EI[0]

                opt_revenue, current_system_state, opt_EO, opt_EI = self.__optimize_revenue(r=r,x=x,w=w,EI=opt_EI,EO=opt_EO,initial_state=state,starting_time_frame=t) 

                if s:
                    self.columns=[' ','Parameters','timeframe #','Initial system state','Current system state','Relocation operation','Relocation number',
                            'Relocation Path','Relocation Cost','Opt_EO','Opt_Revenue','OD matrix','Distance matrix']
                    if not title:
                        results = pd.DataFrame(columns=self.columns)
                        # results.to_csv(current_path + '/results.csv', mode='a')
                        title = True
                    data, relo_operation = self.__save_results(time_frame=t, opt_revenue = opt_revenue, opt_EO = opt_EO, initial_system_state=c, 
                                                current_system_state=current_system_state, r_dict=r_dict, w_dict=w_dict, x_dict=x_dict)

                    # results = results.append(data,ignore_index=True)
                    results.loc[len(results.index)] = data
                    opti_r.append(relo_operation)

            stop_time = time()
            if s:
                # print(f'Satisfied demand over time-frame {t} is {satisfied_demand[0]} out of {sum(sum(self.__origin_destination_matrix[t]))}')
                results.to_csv(self.path,mode=self.mode)
                # mode = 'a' / 'w'
                with open(current_path + '/' + 'results/relocation.json','w') as f:
                    json.dump(opti_r,f)
                    
                if self.__verbose:
                    print(f'The results are saved in {self.path}')
                    print(f'{stop_time - start_time:.4f} seconds elapsed')
                print()
        return self.status

    def __optimize_relocation(self, starting_time_frame: int, initial_state: list) -> list:

        # Create the linear solver with the CBC backend.
        self.solver = pywraplp.Solver.CreateSolver('CBC_MIXED_INTEGER_PROGRAMMING')
        # self.solver = pywraplp.Solver.CreateSolver('SCIP')

        # Create variables
        s = [[self.solver.NumVar(0, self.solver.infinity(), 
            name='s_' + str(t) + '_' + str(i)) if t != 0 else initial_state[i]  for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]

        EI = [[self.solver.NumVar(0, np.double(self.__incoming_demand[t + starting_time_frame][i]),
            'EI_' + str(t) + '_' + str(i)) for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]
        
        EO = [[self.solver.NumVar(0, np.double(self.__outgoing_demand[t + starting_time_frame][i]),
            'EO_' + str(t) + '_' + str(i)) for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]
        
        r = [[self.solver.IntVar(-self.__maximum_relocation, self.__maximum_relocation,
            'r_' + str(t) + '_' + str(i)) for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]
        
        w = [[[self.solver.IntVar(0, self.solver.infinity(),
            'w_' + str(t) + '_' + str(i) + '_' + str(j)) for j in range(self.__zones_number)] for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]

        x = [[[self.solver.IntVar(0, 1,
            'x_' + str(t) + '_' + str(i) + '_' + str(j)) for j in range(self.__zones_number)] for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)]

        # 2. Add constraints for each resource

        # for t in range(self.__look_ahead_horizon):
        #     self.solver.Add(sum(w[t][i] for i in range(self.__zones_number)) <= self.__maximum_relocation,
        #         'max_relocation_' + str(t))

        # for i in range(self.__zones_number):
        #     for t in range(self.__look_ahead_horizon):
        #         self.solver.Add(w[t][i] >= r[t][i], 'max_relocation_nonnegativity_' + str(t) + '_' + str(i))

        for i in range(self.__zones_number):
            for t in range(self.__look_ahead_horizon):
                self.solver.Add(EO[t][i] <= s[t][i] + 0.5 * EI[t][i] + 0.5 *r[t][i],
                    'effective_outgoing_bound_' + str(t) + '_' + str(i))
        # for i in range(self.__zones_number):
        #     for t in range(self.__look_ahead_horizon):
        #         self.solver.Add(EO[t][i] <= s[t][i] + EI[t][i] + r[t][i],
        #             'effective_outgoing_bound_' + str(t) + '_' + str(i))

        for i in range(self.__zones_number):
            self.solver.Add(-r[0][i] <= s[0][i], 'relocation_bound_' + '0' + str(i))

        for i in range(self.__zones_number):
            for t in range(1,self.__look_ahead_horizon):
                self.solver.Add(-r[t][i] <= s[t][i], 'relocation_bound_' + str(t) + '_' + str(i))

        # equality constraints
        for i in range(self.__zones_number):
            for t in range(1,self.__look_ahead_horizon):
                self.solver.Add(s[t][i] == s[t - 1][i] + EI[t - 1][i] - EO[t - 1][i] + r[t - 1][i],
                          'state_transition_' + str(t) + '_' + str(i))

        for t in range(self.__look_ahead_horizon):
            self.solver.Add(sum(r[t][i] for i in range(self.__zones_number)) == 0, 'tot_relocation_' + str(t))

        for t in range(self.__look_ahead_horizon):
            self.solver.Add(sum(s[t][i] for i in range(self.__zones_number)) == self.__vehicle_number,
                'mass_conservation_' + str(t))

        for i in range(self.__zones_number):
            for t in range(self.__look_ahead_horizon):
                self.solver.Add(EI[t][i] == sum(
                    EO[t][j] * self.__origin_destination_probability[t + starting_time_frame][j, i] for j in
                    self.__possible_origins[t + starting_time_frame][i]),
                          'effective_incoming_outgoing_matching_' + str(t) + '_' + str(i))

        for t in range(self.__look_ahead_horizon):
            for i in range(self.__zones_number):
                for j in range(self.__zones_number):
                    self.solver.Add(w[t][i][j] <= self.C_truck*x[t][i][j], 'truck_relocation_' + str(t) + '_' + str(i) + '_' + str(j))

        for i in range(self.__zones_number):
            for t in range(self.__look_ahead_horizon):
                self.solver.Add(sum(w[t][j][i] for j in range(self.__zones_number)) - sum(w[t][i][j] for j in range(self.__zones_number)) == r[t][i],
                'max_relocation_nonnegativity_' + str(t) + '_' + str(i) + '_' + str(j))               

        self.solver.Add(0 == sum(x[t][i][i] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)), 'not_in_zone')

        for t in range(self.__look_ahead_horizon):
            self.solver.Add(sum(x[t][i][j] for i in range(self.__zones_number) for j in range(self.__zones_number)) <= self.N_truck,
            'max_truck_number_' + str(t))

        # for i in range(self.__zones_number):
        #     for t in range(self.__look_ahead_horizon):
        #         self.solver.Add(1 >= sum(x[t][j][i] for j in range(self.__zones_number)),
        #                 'one_trip_to_one_zone_' + str(t) + '_' + str(i))

        self.solver.Add(0 == sum(x[t][0][j] for t in range(self.__look_ahead_horizon) for j in range(self.__zones_number)), '0x')
        self.solver.Add(0 == sum(x[t][i][0] for t in range(1,self.__look_ahead_horizon) for i in range(self.__zones_number)), '0x1')

        # for i in range(self.__zones_number):
        #     for t in range(1,self.__look_ahead_horizon):
        #         self.solver.Add(0 == r[t][i],
        #                 '0r_' + str(t) + '_' + str(i))

#         self.solver.Add(0 == sum(w[t][0][j] for t in range(0,self.__look_ahead_horizon) for j in range(self.__zones_number)), '0w')
#         self.solver.Add(0 == sum(w[t][i][0] for t in range(0,self.__look_ahead_horizon) for i in range(self.__zones_number)), '0w')
        
#         for t in range(0,self.__look_ahead_horizon):
#                 self.solver.Add(0 == r[t][0],
#                         '0r_' + str(t))

        
        # 3. Maximize the objective function
        self.solver.Maximize((self.I)*sum(EO[t][i] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)) 
                            - self.Cm*sum(x[t][i][j]*self.distances[i][j] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) for j in range(self.__zones_number))
                            - self.Cn*sum(w[t][i][j] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) for j in range(self.__zones_number)))
        # self.solver.Maximize(sum(EO[t][i] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)))
        self.solver.SetTimeLimit(TimeLimit)

        # Solve problem
        self.status = self.solver.Solve()
        print(f'Result Status: {ResultStatus[self.status]} over time frame: {starting_time_frame} with lookahead horizon {self.__look_ahead_horizon}')

        # If an optimal solution has been found, print results
        if self.status == pywraplp.Solver.OPTIMAL or self.status == 1:
            s_dict = {'s_' + str(t) + '_' + str(self.zones_id[i]): s[t][i].solution_value() if t != 0 else s[t][i] for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) }
            r_dict = {'r_' + str(t) + '_' + str(self.zones_id[i]): r[t][i].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) }
            # w_dict = {'w_' + str(t) + '_' + str(i): w[t][i].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)}
            w_dict = {'w_' + str(t) + '_' + str(self.zones_id[i]) + '_' + str(self.zones_id[j]): w[t][i][j].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) for j in range(self.__zones_number)}
            x_dict = {'x_' + str(t) + '_' + str(self.zones_id[i]) + '_' + str(self.zones_id[j]): x[t][i][j].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number) for j in range(self.__zones_number)}
            EI_dict = {'EI_' + str(t) + '_' + str(self.zones_id[i]): EI[t][i].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)}
            EO_dict = {'EO_' + str(t) + '_' + str(self.zones_id[i]): EO[t][i].solution_value() for t in range(self.__look_ahead_horizon) for i in range(self.__zones_number)}
            
            if self.__verbose:
                # x_dict = {'x_' + str(t) + '_' + str(i) + '_' + str(j): x[t][i][j].solution_value() for i in range(self.__zones_number) for j in range(self.__zones_number) for t in range(self.__look_ahead_horizon)}
                # print(f' - x : {x_dict}')
                print()
                print(f'The model solved in {self.solver.wall_time()*0.001:.6f} seconds in {self.solver.iterations()} iterations')
                print(f'Optimal value of Revenue = {self.solver.Objective().Value()}')
                print(f'Over time frame: {starting_time_frame}')
                sn = {k: v for k, v in s_dict.items()}
                print(f' - s : {sn}')
                rn = {k: v for k, v in r_dict.items() if v!= 0}
                print(f' - r : {rn}')
                wn = {k: v for k, v in w_dict.items() if v!= 0}
                print(f' - w : {wn}')
                xn = {k: v for k, v in x_dict.items() if v!= 0}
                print(f' - x : {xn}')
                ei = {k: v for k, v in EI_dict.items() if v!= 0}
                print(f' - EI : {ei}')
                eo = {k: v for k, v in EO_dict.items() if v!= 0}
                print(f' - EO : {eo}')
        else:
            return

        return  [[s[t][i].solution_value() if t != 0 else s[t][i] for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)],\
            [[r[t][i].solution_value() for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)], \
            [[[x[t][i][j].solution_value() for j in range(self.__zones_number)] for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)] ,  \
            [[[w[t][i][j].solution_value() for j in range(self.__zones_number)] for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)], \
            [[EI[t][i].solution_value() for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)], \
            [[EO[t][i].solution_value() for i in range(self.__zones_number)] for t in range(self.__look_ahead_horizon)], \
            r_dict, w_dict, x_dict

    def __optimize_revenue(self,r,x,w,EI,EO,initial_state,starting_time_frame):
        s = initial_state
        EI_value = EI
        EO_value = EO

# round the results
        rounded_EO = list(map(round, EO_value))
        rounded_EI = list(map(round, EI_value))
        unbalance = sum(rounded_EO) - sum(rounded_EI)
        if unbalance > 0:
            indexes = self.__get_rounding_indexes(EO_value, rounded_EO, unbalance)
            while unbalance > 0:
                if len(indexes) > 0:
                    ind = indexes.pop(0)
                else:
                    ind = random.sample(range(self.__zones_number), k=1)[0]
                if s[ind] - (rounded_EO[ind] - 1) + rounded_EI[ind] + r[0][ind] >= 0:
                    rounded_EO[ind] -= 1
                    unbalance -= 1
        elif unbalance < 0:
            indexes = self.__get_rounding_indexes(EI_value, rounded_EI, unbalance)
            while unbalance < 0:
                if len(indexes) > 0:
                    ind = indexes.pop(0)
                else:
                    ind = random.sample(range(self.__zones_number), k=1)[0]
                if s[ind] - rounded_EO[ind] + (rounded_EI[ind] - 1) + r[0][ind] >= 0:
                    rounded_EI[ind] -= 1
                    unbalance += 1

        # s = [s[i] - EO_value[i] + EI_value[i] + r[0][i] for i in range(self.__zones_number)]
        s = [s[i] - rounded_EO[i] + rounded_EI[i] + r[0][i] for i in range(self.__zones_number)]
        obj = [(self.I)*sum(rounded_EO[i] for i in range(self.__zones_number)) 
                - self.Cm*sum(x[0][i][j]*self.distances[i][j] for i in range(self.__zones_number) for j in range(self.__zones_number))
                - self.Cn*sum(w[0][i][j] for i in range(self.__zones_number) for j in range(self.__zones_number))]

        total_objective = sum(obj)
        # total_EO.append(sum(rounded_EO))
        return total_objective, s, rounded_EO, rounded_EI

    def __get_rounding_indexes(self,
                            array: list,
                            rounded_array: list,
                            unbalance: int) -> list:
        rounded_indexes = [(i, (rounded_array[i] - array[i])) for i in range(len(array)) if
                        (rounded_array[i] - array[i]) > TOLERANCE]  # identifying indexes of rounded entries
        if unbalance < len(rounded_indexes):
            rounded_indexes.sort(key=lambda x: x[1], reverse=True)
        return list(map(lambda x: x[0], rounded_indexes)) 
    
    def __simulate_revenue(self, s: list, r: list, time_frame: int, x_dict: dict):
        t = time_frame
        states = [s]
        # r_zero = [0]*self.__zones_number
        # for i in range(1,len(r)):
        #     r[i] = r_zero
        flows = self.system_simulator.get_flow_by_time_frame(t)
        s_r = [states[0][i] + min(0, r[0][i]) for i in range(self.__zones_number)]
        outgoing = [0] * self.__zones_number
        revenue = 0
        flows = [flows[:int(len(flows) / 2)], flows[int(len(flows) / 2):]]
        for f in range(2):
            s_r_copy = s_r.copy()
            for start_zone, arrival_zone in flows[f]:
                if s_r_copy[start_zone] > 0:
                    s_r[start_zone] -= 1
                    s_r[arrival_zone] += 1
                    s_r_copy[start_zone] -= 1
                    outgoing[start_zone] += 1
                    revenue += (self.I)
            if f == 0:
                s_r = [s_r[i] + max(0, r[0][i]) for i in range(self.__zones_number)]
                for key in x_dict:
                    if x_dict[key] != 0:
                        z = key.split('_')
                        if int(z[1]) == 0:
                            revenue -= self.Cm*self.distances[int(z[2])][int(z[3])]
                for i in range(self.__zones_number):
                    revenue -= max(0, r[0][i]) * self.Cn

        total_revenue = round(revenue,2)
        states.append(s_r)
        return states[1], states, outgoing, total_revenue

    def __save_results(self, time_frame, opt_revenue, opt_EO, initial_system_state, 
                        current_system_state, r_dict, w_dict, x_dict):

        path = []
        for key in x_dict:
            if x_dict[key] != 0:
                p = key.split('_')
                if p[1] == str(0):
                    path.append((p[2],p[3]))

        opera = []
        num = 0
        # for key,value in r_dict.items():
        #     if r_dict[key] < 0:
        #         s = key.split('_')
        #         opera.append((s[2],value))
            # elif r_dict[key] > 0:
            #     e = key.split('_')
            #     opera.append((e[2],value))
        for key,value in w_dict.items():
            if w_dict[key] != 0:
                s = key.split('_')
                if s[1] == str(0):
                    opera.append((s[2],value*-1))
                    opera.append((s[3],value))
                    num += value
        cost = num*self.Cn + len(path)*self.Cm

        now = time()
        data=[now, self.parameters, time_frame, initial_system_state, current_system_state, opera, num, path, cost,\
            [sum(opt_EO),opt_EO], opt_revenue, self.__origin_destination_matrix[time_frame], self.distances]
        relo_operation = {'time_frame': time_frame, 'relo_operation': opera}
        # print(opera)
        return data, relo_operation

if __name__ == '__main__':
    overall_time_start = time()

    paras = {'N_scooter': 160, 'N_truck': 5, 'C_truck': 10, 'Incoming': 10, 'Cm': 1, 'Cn': 0.1}
    # C_truck - capacity of each truck
    # Incoming - incoming per trip
    # Cm - relocation cost / unit mileage cost
    # Cn - relocation cost / unit number of scooters cost

    s = SystemGeneration(N_zones=8,coord=None,raw_data=None)
    raw_data, coords = s.generate_rawdata()
    distances = np.load(current_path + '/' + 'distances.npy')
    print(raw_data)
    print(distances)

    s.generate_ODmatrix(od=1)
    # distances = np.load(current_path + '/distances.npy')
    od_mat = np.load(current_path + '/od_mat.npy').astype(np.uint8)
    tfs = SystemSimulator(origin_destination_matrix=od_mat)
    
    solution = 'simulation'
    solution = 'optimization'
    file_path = current_path + '/results - {}.csv'.format(solution)

    o = Optimizer(parameters = paras, origin_destination_matrix=od_mat, distances=distances, optimization_horizon=1,
        look_ahead_horizon=1, system_simulator=tfs, verbose=True, file_path =file_path, mode = 'a')
    o.run_optimizer(solution)

# # test - truck
#     file_path = current_path + '/results/test - trucks.csv'
#     for i in range(5):
#         for j in range(5):
#             paras['N_truck'] = i
#             paras['C_truck'] = j*10
#             Optimizer(parameters = paras, origin_destination_matrix=od_mat, distances=distances, optimization_horizon=2,
#             look_ahead_horizon=3, system_simulator=tfs, verbose=True, file_path =file_path)

# # test - cost
#     file_path = current_path + '/results/test - costs.csv'
#     for i in range(5):
#         for j in range(5):
#             paras['Cm'] = 90+5*i
#             paras['Cn'] = 4+0.5*j
#             Optimizer(parameters = paras, origin_destination_matrix=od_mat, distances=distances, optimization_horizon=2,
#             look_ahead_horizon=3, system_simulator=tfs, verbose=True, file_path =file_path)

# # test - timehorizon
#     file_path = current_path + '/results/test - timehorizons.csv'
#     for i in range(1,5):
#         for j in range(1,5):
#             to = i
#             tl = j
#             Optimizer(parameters = paras, origin_destination_matrix=od_mat, distances=distances, optimization_horizon=to,
#              look_ahead_horizon=tl, system_simulator=tfs, file_path=file_path, verbose=True)
    

    overall_time_stop= time()
    print(f'Overall time: {overall_time_stop-overall_time_start:.4f} seconds')