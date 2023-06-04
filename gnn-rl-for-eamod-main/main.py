from __future__ import print_function
import argparse
import os
import gurobipy as gp
from tqdm import trange
import numpy as np
from src.algos.pax_flows_solver import PaxFlowsSolver
from src.algos.reb_flow_solver import RebalFlowSolver
import torch
import json
import os
import wandb
import pickle
import time
import math

from src.envs.amod_env import Scenario, AMoD
from src.algos.a2c_gnn import A2C
from src.algos.a2c_gnn_2 import A2C as A2C_2
from src.misc.utils import dictsum

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

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--equal_distr_baseline', type=bool, default=False,
                    help='activates the equal distribution baseline.')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--T', type=int, default=10, metavar='N',
                    help='Time horizon for the A2C')
parser.add_argument('--lr_a', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the actor')
parser.add_argument('--lr_c', type=float, default=1e-3, metavar='N',
                    help='Learning rate for the critic')
parser.add_argument('--grad_norm_clip_a', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the actor')
parser.add_argument('--grad_norm_clip_c', type=float, default=0.5, metavar='N',
                    help='Gradient norm clipping for the critic')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
lr_a = args.lr_a
lr_c = args.lr_c
grad_norm_clip_a = args.grad_norm_clip_a
grad_norm_clip_c = args.grad_norm_clip_c
use_equal_distr_baseline = args.equal_distr_baseline
seed = args.seed
test = args.test
T = args.T

problem_folder = 'Toy'
file_path = os.path.join('data', problem_folder, 'scenario_test_3_2.json')
experiment = 'training_' + problem_folder+ '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + file_path
energy_dist_path = os.path.join('data', problem_folder,  'energy_distance_3x2.npy')
scenario = create_scenario(file_path, energy_dist_path)
env = AMoD(scenario)
scale_factor = 0.01
scale_price = 0.1
model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
model_2 = A2C_2(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
tf = env.tf

# if args.toy:
#     problem_folder = 'Toy'
#     file_path = os.path.join('data', problem_folder, 'scenario_test_6_1x2_flip.json')
#     experiment = 'training_' + problem_folder+ '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + file_path
#     energy_dist_path = os.path.join('data', problem_folder,  'energy_distance_1x2.npy')
#     scenario = create_scenario(file_path, energy_dist_path)
#     env = AMoD(scenario)
#     scale_factor = 0.01
#     scale_price = 0.1
#     model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
#     # model.load_checkpoint(path=f'saved_files/ckpt/{problem_folder}/a2c_gnn_50000.pth')
#     tf = env.tf
# else:
#     # problem_folder = 'SF_5_clustered'
#     # file_path = os.path.join('data', problem_folder,  'SF_5_short.json')
#     # problem_folder = 'NY/ClusterDataset1'
#     # file_path = os.path.join('data', problem_folder,  'd1.json')
#     problem_folder = 'NY_5'
#     file_path = os.path.join('data', problem_folder,  'NY_5_day.json')
#     experiment = 'training_' + file_path + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T)
#     energy_dist_path = os.path.join('data', problem_folder, 'energy_distance.npy')
#     scenario = create_scenario(file_path, energy_dist_path)
#     env = AMoD(scenario)
#     # Initialize A2C-GNN
#     # # NY
#     # scale_factor = 0.01
#     # scale_price = 0.1
#     # SF
#     # scale_factor = 0.00001
#     # scale_price = 0.1
#     # NY 5 
#     scale_factor = 0.0001
#     scale_price = 0.1
#     model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed, scale_factor=scale_factor, scale_price=scale_price).to(device)
#     # model.load_checkpoint(path=f'saved_files/ckpt/{problem_folder}/a2c_gnn_40000.pth')
#     tf = env.tf

if use_equal_distr_baseline:
    experiment = 'uniform_distr_baseline_' + file_path + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T)
if test:
    experiment += "_test_evaluation"
experiment += "_RL_approach_constraint"

# set Gurobi environment Justin
# gurobi_env = gp.Env(empty=True)
# gurobi = "Justin"
# gurobi_env.setParam('WLSACCESSID', '82115472-a780-40e8-9297-b9c92969b6d4')
# gurobi_env.setParam('WLSSECRET', '0c069810-f45f-4920-a6cf-3f174425e641')
# gurobi_env.setParam('LICENSEID', 844698)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# # set Gurobi environment Karthik2
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
        "episodes": args.max_episodes,
        "number_vehicles_per_node_init": env.G.nodes[(0,1)]['accInit'],
        "charging_stations": list(env.scenario.charging_stations),
        "charging_station_capacities": list(env.scenario.cars_per_station_capacity),
        "learning_rate_actor": lr_a,
        "learning_rate_critic": lr_c,
        "gradient_norm_clip_actor": grad_norm_clip_a,
        "gradient_norm_clip_critic": grad_norm_clip_c,
        "scale_factor": scale_factor,
        "scale_price": scale_price,
        "time_horizon": T,
        "episode_length": env.tf,
        "seed": seed,
        "charge_levels_per_timestep": env.scenario.charge_levels_per_charge_step, 
        "licence": gurobi,
      })


################################################
#############Training and Eval Loop#############
################################################
n_episodes = args.max_episodes #set max number of training episodes
T = tf #set episode length
epochs = trange(n_episodes) #epoch iterator
best_reward = -10000
best_model = None
if test:
    rewards_np = np.zeros(n_episodes)
    served_demands_np = np.zeros(n_episodes)
    charging_costs_np = np.zeros(n_episodes)
    rebal_costs_np = np.zeros(n_episodes)
    epoch_times = np.zeros(n_episodes)
else:
    model.train() #set model in train mode
total_demand_per_spatial_node = np.zeros(env.number_nodes_spatial)
for region in env.nodes_spatial:
    for destination in env.nodes_spatial:
        for t in range(env.tf):
            total_demand_per_spatial_node[region] += env.demand[region,destination][t]
k = 250
grad_prop = True
for i_episode in epochs:
    desired_accumulations_spatial_nodes = np.zeros(env.scenario.spatial_nodes)
    bool_random_random_demand = not test # only use random demand during training
    obs = env.reset(bool_random_random_demand) #initialize environment
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
    time_start = time.time()
    action_tracker = {}
    for step in range(T):
        # take matching step (Step 1 in paper)
        # if ((i_episode < k)):
            # linear optimization
        if step == 0 and i_episode == 0:
            # initialize optimization problem in the first step
            pax_flows_solver = PaxFlowsSolver(env=env,gurobi_env=gurobi_env)
        else:
            pax_flows_solver.update_constraints()
            pax_flows_solver.update_objective()
        _, paxreward, done, info_pax = env.pax_step(pax_flows_solver=pax_flows_solver, episode=i_episode)
        episode_reward += paxreward
        # else:
        #     # RL 1
        #     if ((k < i_episode < 2 * k) or (3 * k < i_episode < 4 * k) or (5 * k < i_episode < 6 * k) or (7 * k < i_episode < 8 * k) or (9 * k < i_episode < 10 * k) or (11 * k < i_episode < 12 * k) or (13 * k < i_episode < 14 * k) or (15 * k < i_episode < 16 * k)):
        #         for param in model_2.parameters():
        #             param.requires_grad = True
        #             grad_prop = True
        #     else:
        #         for param in model_2.parameters():
        #             param.requires_grad = False
        #             grad_prop = False
        #     action_rl_one = model_2.select_action()
        #     # calculate a flow-tuple of length number of edges by calculating 
        #     # difference between node_destination and node_origin
        #     flow = []
        #     edges = env.edges
        #     for node_destination_idx in range(env.number_nodes):
        #         for node_origin_idx in range(env.number_nodes):
        #             edge = (env.nodes[node_origin_idx], env.nodes[node_destination_idx])
        #             if edge in edges:
        #                 total_acc = sum(env.acc[n][env.time] for n in env.nodes)
        #                 flow.append(total_acc * (action_rl_one[node_destination_idx] - action_rl_one[node_origin_idx]))
        #     _ , paxreward, done, info_pax = env.pax_step(paxAction=flow, episode=i_episode)
        #     episode_reward += paxreward

        # # use GNN-RL policy (Step 2 in paper)
        # if ((k < i_episode < 2 * k) or (3 * k < i_episode < 4 * k) or (5 * k < i_episode < 6 * k) or (7 * k < i_episode < 8 * k) or (9 * k < i_episode < 10 * k) or (11 * k < i_episode < 12 * k) or (13 * k < i_episode < 14 * k) or (15 * k < i_episode < 16 * k)):
        #     for param in model.parameters():
        #         param.requires_grad = False
        #         grad_prop = False
        # else:
        #     for param in model.parameters():
        #         param.requires_grad = True
        #         grad_prop = True
        action_rl = model.select_action()           

        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        total_idle_acc = sum(env.acc[n][env.time+1] for n in env.nodes)
        desired_acc = {env.nodes[i]: int(action_rl[i] *total_idle_acc) for i in range(env.number_nodes)} # over nodes
        action_tracker[step] = desired_acc
        total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
        missing_cars = total_idle_acc - total_desiredAcc
        most_likely_node = np.argmax(action_rl)
        if missing_cars != 0:
            desired_acc[env.nodes[most_likely_node]] += missing_cars   
            total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
        assert abs(total_desiredAcc - total_idle_acc) < 1e-5
        for n in env.nodes:
            assert desired_acc[n] >= 0
        for n in env.nodes:
            desired_accumulations_spatial_nodes[n[0]] += desired_acc[n]

        # solve minimum rebalancing distance problem (Step 3 in paper)
        if step == 0 and i_episode == 0:
            # initialize optimization problem in the first step
            rebal_flow_solver = RebalFlowSolver(env=env, desiredAcc=desired_acc, gurobi_env=gurobi_env)
        else:
            rebal_flow_solver.update_constraints(desired_acc, env)
            rebal_flow_solver.update_objective(env)
        rebAction = rebal_flow_solver.optimize()
        # if (i_episode % 1000 == 0):
        #     for i in range(len(env.edges)):
        #         print(str(env.edges[i]) + ", rebAction: " + str(rebAction[i]))
        # currently, rebAction is not returning a rebalancing action - hence, there is an error with rebal_flow_solver

        # Take action in environment
        new_obs, rebreward, done, info_reb = env.reb_step(rebAction)
        episode_reward += rebreward
        # Store the transition in memory
        model.rewards.append(paxreward + rebreward)
        # track performance over episode
        episode_served_demand += info_pax['served_demand']
        episode_rebalancing_cost += info_reb['rebalancing_cost']
        # stop episode if terminating conditions are met
        if done:
            break
    # perform on-policy backprop
    if not use_equal_distr_baseline:
        if grad_prop:
            a_loss, v_loss, mean_value, mean_concentration, mean_std, mean_log_prob, std_log_prob = model.training_step()

    # Send current statistics to screen was episode_reward, episode_served_demand, episode_rebalancing_cost
    epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")
    # Send current statistics to wandb
    for spatial_node in range(env.scenario.spatial_nodes):
        wandb.log({"Episode": i_episode+1, f"Desired Accumulation {spatial_node}": desired_accumulations_spatial_nodes[spatial_node]})
        wandb.log({"Episode": i_episode+1, f"Total Demand {spatial_node}": total_demand_per_spatial_node[spatial_node]})
        if total_demand_per_spatial_node[spatial_node] > 0:
            wandb.log({"Episode": i_episode+1, f"Desired Acc. to Total Demand ratio {spatial_node}": desired_accumulations_spatial_nodes[spatial_node]/total_demand_per_spatial_node[spatial_node]})
    # Checkpoint best performing model
    if episode_reward > best_reward:
        print("Saving best model.")
        if (i_episode >= 10000):
            for step in action_tracker:
                print("Time step: " + str(step) + ", desired cars at nodes after policy's rebalancing action: " + str(action_tracker[step]))
        model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn.pth")
        best_model = model
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn.pth")
        with open(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p", "wb") as file:
            pickle.dump(env.acc_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/n_charging_vehicles_spatial.p", "wb") as file:
            pickle.dump(env.n_charging_vehicles_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_charging_vehicles_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/n_rebal_vehicles_spatial.p", "wb") as file:
            pickle.dump(env.n_rebal_vehicles_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_rebal_vehicles_spatial.p")
        with open(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p", "wb") as file:
            pickle.dump(env.n_customer_vehicles_spatial, file)
        wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial.p")
        best_reward = episode_reward
        best_rebal_cost = episode_rebalancing_cost
        best_served_demand  = episode_served_demand
    if test:
        rewards_np[i_episode] = episode_reward
        served_demands_np[i_episode] = episode_served_demand
        rebal_costs_np[i_episode] = episode_rebalancing_cost
        epoch_times[i_episode] = time.time()-time_start
    else:
        wandb.log({"Episode": i_episode+1, "Reward": episode_reward, "Best Reward:": best_reward, "ServedDemand": episode_served_demand, "Best Served Demand": best_served_demand, 
        "Reb. Cost": episode_rebalancing_cost, "Best Reb. Cost": best_rebal_cost, "Spatial Reb. Cost": -rebreward,
        "Actor Loss": a_loss, "Value Loss": v_loss, "Mean Value": mean_value, "Mean Concentration": mean_concentration, "Mean Std": mean_std, "Mean Log Prob": mean_log_prob, "Std Log Prob": std_log_prob})
        # regularly safe model
        if i_episode % 10000 == 0:
            model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_{i_episode}.pth")
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_{i_episode}.pth")
            with open(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial_{i_episode}.p", "wb") as file:
                pickle.dump(env.acc_spatial, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/acc_spatial_{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/n_charging_vehicles_spatial_{i_episode}.p", "wb") as file:
                pickle.dump(env.n_charging_vehicles_spatial, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_charging_vehicles_spatial_{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/n_rebal_vehicles_spatial_{i_episode}.p", "wb") as file:
                pickle.dump(env.n_rebal_vehicles_spatial, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_rebal_vehicles_spatial_{i_episode}.p")
            with open(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial_{i_episode}.p", "wb") as file:
                pickle.dump(env.n_customer_vehicles_spatial, file)
            wandb.save(f"./{args.directory}/ckpt/{problem_folder}/n_customer_vehicles_spatial_{i_episode}.p")
if test:
    print(rewards_np)
    wandb.log({"AVG Reward ": rewards_np.mean(), "Std Reward ": rewards_np.std(), "AVG Satisfied Demand ": served_demands_np.mean(), "AVG Rebalancing Cost": episode_rebalancing_cost.mean(), "AVG Epoch Time": epoch_times.mean()})
model.save_checkpoint(path=f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_final.pth")
wandb.save(f"./{args.directory}/ckpt/{problem_folder}/a2c_gnn_final.pth")
wandb.finish()


print("Evaluating best model with greedy mean action selection from Dirichlet distribution") 


desired_accumulations_spatial_nodes = np.zeros(env.scenario.spatial_nodes)
bool_random_random_demand = False # only use random demand during training
obs = env.reset(bool_random_random_demand) #initialize environment
episode_reward = 0
episode_served_demand = 0
episode_rebalancing_cost = 0
time_start = time.time()
action_tracker = {}
for step in range(T):
    # take matching step (Step 1 in paper)
    if step == 0 and i_episode == 0:
        # initialize optimization problem in the first step
        pax_flows_solver = PaxFlowsSolver(env=env,gurobi_env=gurobi_env)
    else:
        pax_flows_solver.update_constraints()
        pax_flows_solver.update_objective()
    _, paxreward, done, info_pax = env.pax_step(pax_flows_solver=pax_flows_solver, episode=i_episode)
    episode_reward += paxreward
   
    # use GNN-RL policy (Step 2 in paper)
    action_rl = best_model.select_action(eval_mode=True)  # vanilla GCN
    # action_rl = best_model.select_action_MPNN(eval_mode=True)  # MPNN
    # action_rl = best_model.select_action_GAT(eval_mode=True)  # GAT
    
    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
    total_idle_acc = sum(env.acc[n][env.time+1] for n in env.nodes)
    desired_acc = {env.nodes[i]: int(action_rl[i] *total_idle_acc) for i in range(env.number_nodes)} # over nodes
    action_tracker[step] = desired_acc
    total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
    missing_cars = total_idle_acc - total_desiredAcc
    most_likely_node = np.argmax(action_rl)
    if missing_cars != 0:
        desired_acc[env.nodes[most_likely_node]] += missing_cars   
        total_desiredAcc = sum(desired_acc[n] for n in env.nodes)
    assert abs(total_desiredAcc - total_idle_acc) < 1e-5
    for n in env.nodes:
        assert desired_acc[n] >= 0
    for n in env.nodes:
        desired_accumulations_spatial_nodes[n[0]] += desired_acc[n]

    # solve minimum rebalancing distance problem (Step 3 in paper)
    if step == 0 and i_episode == 0:
        # initialize optimization problem in the first step
        rebal_flow_solver = RebalFlowSolver(env=env, desiredAcc=desired_acc, gurobi_env=gurobi_env)
    else:
        rebal_flow_solver.update_constraints(desired_acc, env)
        rebal_flow_solver.update_objective(env)
    rebAction = rebal_flow_solver.optimize()
       
    # Take action in environment
    new_obs, rebreward, done, info_reb = env.reb_step(rebAction)
    episode_reward += rebreward
    # Store the transition in memory
    best_model.rewards.append(paxreward + rebreward)
    # track performance over episode
    episode_served_demand += info_pax['served_demand']
    episode_rebalancing_cost += info_reb['rebalancing_cost']
    # stop episode if terminating conditions are met
    if done:
        break

# Send current statistics to screen was episode_reward, episode_served_demand, episode_rebalancing_cost
print(f"Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")

print("done")