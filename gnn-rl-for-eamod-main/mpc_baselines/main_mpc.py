import sys
import argparse
sys.path.append('./src')
from src.envs.amod_env import Scenario, AMoD #, Star2Complete
from src.misc.utils import mat2str, dictsum
from MPC import MPC
import time
import os
import subprocess
from collections import defaultdict
import numpy as np
import gurobipy as gp
import json
import wandb
import pickle
import math

def create_scenario(json_file_path, energy_file_path, seed=10):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    spatial_nodes = data['spatialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_levels_per_charge_step = data['chargeLevelsPerChargeStep']
    chargers = data['chargeLocations']
    cars_per_station_capacity = data['carsPerStationCapacity']
    p_energy = data["energy_prices"]

    t = 0
    for element in p_energy:
        element = (math.ceil(10 * element * np.sin(t/(2*np.pi))))/(10.0)
        t += 1

    time_granularity = data["timeGranularity"]
    operational_cost_per_timestep = data['operationalCostPerTimestep']

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, cars_per_station_capacity = cars_per_station_capacity, number_charge_levels=number_charge_levels, charge_levels_per_charge_step=charge_levels_per_charge_step, 
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc, p_energy=p_energy, time_granularity=time_granularity, operational_cost_per_timestep=operational_cost_per_timestep)
    return scenario

parser = argparse.ArgumentParser(description='A2C-GNN')

parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--mpc_horizon', type=int, default=60, metavar='N',
                    help='MPC horizon (default: 60)')
parser.add_argument('--subproblem', type=int, default=0, metavar='N',
                    help='which subproblem to run (default: 0)')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
args = parser.parse_args()

mpc_horizon = args.mpc_horizon

if args.toy:
    problem_folder = 'Toy'
    file_path = os.path.join('.', 'data', problem_folder, 'scenario_test_6_1x2_flip.json')
    experiment = file_path +  '_mpc_horizon_' + str(mpc_horizon)
    energy_dist_path = os.path.join('.', 'data', problem_folder,  'energy_distance_1x2.npy')
    scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(scenario)
    tf = env.tf
else:
    # problem_folder = 'SF_5_clustered'
    # file_path = os.path.join('..', 'data', problem_folder, 'SF_5_short.json')
    problem_folder = 'NY_5'
    file_path = os.path.join('.', 'data', problem_folder, 'NY_5.json')
    experiment = file_path +  '_mpc_horizon_' + str(mpc_horizon)
    energy_dist_path = os.path.join('.', 'data', problem_folder, 'energy_distance.npy')
    test_scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(test_scenario)
    tf = env.tf
print('mpc_horizon', mpc_horizon, 'episodeLength', tf)
experiment += "_RL_approach_constraint"

# set Gurobi environment Justin
# gurobi_env = gp.Env(empty=True)
# gurobi = "Justin"
# gurobi_env.setParam('WLSACCESSID', '82115472-a780-40e8-9297-b9c92969b6d4')
# gurobi_env.setParam('WLSSECRET', '0c069810-f45f-4920-a6cf-3f174425e641')
# gurobi_env.setParam('LICENSEID', 844698)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set Gurobi environment Karthik2
gurobi_env = gp.Env(empty=True)
gurobi = "Karthik2"
gurobi_env.setParam('WLSACCESSID', 'bc0f99a5-8537-45c3-89d9-53368d17e080')
gurobi_env.setParam('WLSSECRET', '6dddd313-d8d4-4647-98ab-d6df872c6eaa')
gurobi_env.setParam('LICENSEID', 799870)
gurobi_env.setParam("OutputFlag",0)
gurobi_env.start()

# set up wandb
wandb.init(
      # Set the project where this run will be logged
      project='e-amod', 
      # pass a run name 
      name=experiment, 
      # Track hyperparameters and run metadata
      config={
      "number_chargelevels": env.scenario.number_charge_levels,
      "number_spatial_nodes": env.scenario.spatial_nodes,
      "dataset": file_path,
      "mpc_horizon": mpc_horizon,
      "number_vehicles_per_node_init": env.G.nodes[(0,1)]['accInit'],
      "charging_stations": list(env.scenario.charging_stations),
      "licence": gurobi,
      })

opt_rew = []
# obs = env.reset() # TODO: determine if we should do this
mpc = MPC(env, gurobi_env, mpc_horizon)
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0
t_0 = time.time()
time_list = []
while(not done):
    time_i_start = time.time()
    paxAction, rebAction = mpc.MPC_exact() 
    time_i_end = time.time()
    t_i = time_i_end - time_i_start
    time_list.append(t_i)
    if (env.tf <= env.time + mpc_horizon):
        timesteps = range(mpc_horizon)
    else:
        timesteps = [0]
    for t in timesteps:
        obs, reward1, done, info = env.pax_step(paxAction[t], gurobi_env)    
        obs, reward2, done, info = env.reb_step(rebAction[t])
        opt_rew.append(reward1+reward2) 
        served += info['served_demand']
        rebcost += info['rebalancing_cost']
        opcost += info['operating_cost']
        revenue += info['revenue'] 
print(f'MPC: Reward {sum(opt_rew)}, Revenue {revenue},Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')
# Send current statistics to wandb
wandb.log({"Reward": sum(opt_rew), "ServedDemand": served, "Reb. Cost": rebcost})
wandb.log({"Reward": sum(opt_rew), "ServedDemand": served, "Reb. Cost": rebcost, "Avg.Time": np.array(time_list).mean()})
with open(f"./saved_files/ckpt/{problem_folder}/acc.p", "wb") as file:
    pickle.dump(env.acc, file)
wandb.save(f"./saved_files/ckpt/{problem_folder}/acc.p")
wandb.save(f"./saved_files/ckpt/{problem_folder}/satisfied_demand.p")
wandb.finish()
wandb.finish()
