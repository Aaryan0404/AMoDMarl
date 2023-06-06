from __future__ import print_function
import argparse
from cmath import inf
import os
import gurobipy as gp
from tqdm import trange
import numpy as np
import torch
import json
import os
import wandb

from src.envs.amod_env import Scenario, AMoD
from src.algos.a2c_gnn import A2C
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum

def create_scenario(json_file_path, energy_file_path, seed=10):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    spatial_nodes = data['spacialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_time_per_level = data['chargeTime']
    chargers = []
    for node in range(spatial_nodes):
        chargers.append(node)

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, number_charge_levels=number_charge_levels, charge_time=charge_time_per_level, 
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc)
    return scenario

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train(subproblem=0, episodes=20, seed=10, test=False):
    # subproblem=0
    # episodes=5000
    # seed=10
    # test=False
    with wandb.init() as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        lr_a = config.lr_a
        lr_c = config.lr_c
        grad_norm_clip_a = 0.5
        grad_norm_clip_c = 0.5
        T = config.T
        seed = config.seed

        training_environments = []
        problem_folder = 'Toy'
        file_path = os.path.join('data', problem_folder, 'scenario_test_3_2.json')
        energy_dist_path = os.path.join('data', problem_folder,  'energy_distance_3x2.npy')
        scenario = create_scenario(file_path, energy_dist_path, seed=seed)
        env = AMoD(scenario, beta=0.5)
        training_environments.append(env)
        
        # Initialize A2C-GNN
        model = A2C(env=env, T=T, lr_a=lr_a, lr_c=lr_c, grad_norm_clip_a=grad_norm_clip_a, grad_norm_clip_c=grad_norm_clip_c, seed=seed).to(device)
        tf = env.tf

        # set Gurobi environment
        gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
        gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
        gurobi_env.setParam('LICENSEID', 799876)
        gurobi_env.setParam("OutputFlag",0)
        gurobi_env.start()


        if not test:
            #######################################
            #############Training Loop#############
            #######################################

            log = {'train_reward': [], 
                'train_served_demand': [], 
                'train_reb_cost': []}
            train_episodes = episodes #set max number of training episodes
            T = tf #set episode length
            epochs = trange(train_episodes)
            best_reward = -inf
            model.train() #set model in train mode

            for i_episode in epochs:
                reward_over_subproblems = 0
                served_demand_over_subproblems = 0
                reb_cost_over_subproblems = 0

                for env_idx in range(1):
                    env = training_environments[env_idx]
                    model.set_env(env)
                    obs = env.reset() #initialize environment
                    episode_reward = 0
                    episode_served_demand = 0
                    episode_rebalancing_cost = 0
                    for step in range(T):
                        # take matching step (Step 1 in paper)
                        obs, paxreward, done, info = env.pax_step(gurobi_env=gurobi_env)
                        episode_reward += paxreward
                        
                        # use GNN-RL policy (Step 2 in paper)
                        action_rl = model.select_action(obs)
                        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                        total_acc = sum(env.acc[n][env.time+1] for n in env.nodes)
                        desiredAcc = {env.nodes[i]: int(action_rl[i] *total_acc) for i in range(env.number_nodes)}
                        total_desiredAcc = sum(desiredAcc[n] for n in env.nodes)
                        missing_cars = total_acc - total_desiredAcc
                        most_likely_node = np.argmax(action_rl)
                        if missing_cars != 0:
                            desiredAcc[env.nodes[most_likely_node]] += missing_cars   
                            total_desiredAcc = sum(desiredAcc[n] for n in env.nodes)
                        assert total_desiredAcc == total_acc
                        for n in env.nodes:
                            assert desiredAcc[n] >= 0
                            
                        # solve minimum rebalancing distance problem (Step 3 in paper)
                        rebAction = solveRebFlow(env=env, desiredAcc=desiredAcc, gurobi_env=gurobi_env)
                        # Take action in environment
                        new_obs, rebreward, done, info = env.reb_step(rebAction)
                        episode_reward += rebreward
                        # Store the transition in memory
                        model.rewards.append(paxreward + rebreward)
                        # new metrics, track performance over subproblems
                        reward_over_subproblems += paxreward + rebreward
                        served_demand_over_subproblems += info['served_demand']
                        reb_cost_over_subproblems += info['rebalancing_cost']
                        # track performance over episode
                        episode_served_demand += info['served_demand']
                        episode_rebalancing_cost += info['rebalancing_cost']
                        # stop episode if terminating conditions are met
                        if done:
                            break
                # perform on-policy backprop
                a_loss, v_loss, mean_value = model.training_step()

                # Send current statistics to screen was episode_reward, episode_served_demand, episode_rebalancing_cost
                epochs.set_description(f"Episode {i_episode+1} | Reward: {reward_over_subproblems:.2f} | ServedDemand: {served_demand_over_subproblems:.2f} | Reb. Cost: {reb_cost_over_subproblems:.2f}")
                # Send current statistics to wandb
                wandb.log({"Episode": i_episode+1, "Reward": reward_over_subproblems, "ServedDemand": served_demand_over_subproblems, "Reb. Cost": reb_cost_over_subproblems, "Subproblem": env_idx, "Actor Loss": a_loss, "Value Loss": v_loss, "Mean Value": mean_value})

                if reward_over_subproblems > best_reward:
                    model.save_checkpoint(path=f"./saved_files/ckpt/{problem_folder}/a2c_gnn.pth")
                    wandb.save(f"./saved_files/ckpt/{problem_folder}/a2c_gnn.pth")
                    best_reward = reward_over_subproblems
                    wandb.log({"Best_Reward": reward_over_subproblems})
                # Log KPIs
                log['train_reward'].append(episode_reward)
                log['train_served_demand'].append(episode_served_demand)
                log['train_reb_cost'].append(episode_rebalancing_cost)
                model.log(log, path=f"./saved_files/rl_logs/{problem_folder}/a2c_gnn.pth")
        else:
            # Load pre-trained model
            model.load_checkpoint(path=f'saved_files/ckpt/{problem_folder}/a2c_gnn.pth')
            test_episodes = episodes #set max number of training episodes
            T = tf #set episode length
            epochs = trange(test_episodes)
            
            #Initialize lists for logging
            log = {'test_reward': [], 
                'test_served_demand': [], 
                'test_reb_cost': []}
            best_reward = -np.inf #set best reward
            for episode in epochs:
                episode_reward = 0
                episode_served_demand = 0
                episode_rebalancing_cost = 0
                obs = env.reset()
                done = False
                while(not done):
                    # take matching step (Step 1 in paper)
                    obs, paxreward, done, info = env.pax_step(gurobi_env=gurobi_env)
                    episode_reward += paxreward
                    # use GNN-RL policy (Step 2 in paper)
                    action_rl = model.select_action(obs)
                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {env.nodes[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.nodes))}
                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(env=env, desiredAcc=desiredAcc, gurobi_env=gurobi_env)
                    # Take action in environment
                    new_obs, rebreward, done, info = env.reb_step(rebAction)
                    episode_reward += rebreward
                    # track performance over episode
                    episode_served_demand += info['served_demand']
                    episode_rebalancing_cost += info['rebalancing_cost']
                # Send current statistics to screen
                epochs.set_description(f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}")
                # Send current statistics to wandb
                wandb.log({"Episode": episode+1, "Reward": episode_reward, "ServedDemand": episode_served_demand, "Reb. Cost": episode_rebalancing_cost})
                
                # Checkpoint best performing model
                if episode_reward > best_reward:
                    print("Saving best model.")
                    model.save_checkpoint(path=f"./ckpt/{problem_folder}/a2c_gnn_result.pth")
                    wandb.save(f"./ckpt/{problem_folder}/a2c_gnn_result.pth")
                    best_reward = episode_reward
                    wandb.log({"Best Reward": best_reward})
                # Log KPIs
                log['test_reward'].append(episode_reward)
                log['test_served_demand'].append(episode_served_demand)
                log['test_reb_cost'].append(episode_rebalancing_cost)
                model.log(log, path=f"./rl_logs/{problem_folder}/a2c_gnn.pth")
        # wandb.summary['max_reward'] = best_rewards.max()
        wandb.summary['max_reward'] = best_reward
        wandb.summary['number_chargelevels'] = env.scenario.number_charge_levels
        wandb.summary['number_spatial_nodes'] = env.scenario.spatial_nodes
        wandb.summary['episodes'] = episodes
        wandb.summary['number_vehicles_per_node_init'] = env.G.nodes[(0,0)]['accInit']
        wandb.summary['charging_stations'] = list(env.scenario.charging_stations)
        wandb.finish()
        print("done")
            
train()
