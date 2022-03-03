import gym
import argparse
import os
import d4rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List
import numba


@torch.no_grad()
def divide2trajectories(data):
    length = data["terminals"].shape[0]
    rewards = []
    observations = []
    actions = []
    start_idx = 0
    for i in range(length):
        if data["terminals"][i] == True:
            rewards.append(data["rewards"][start_idx:i])
            observations.append(data["observations"][start_idx:i])
            actions.append(data["actions"][start_idx:i])
            start_idx=i+1
    results = dict(rewards=rewards, observations=observations, actions=actions)
    return results

def dict_to_tensor(inputs, device):
    return {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in inputs.items()}

@torch.no_grad()
@torch.jit.script
def get_returns(starts: torch.Tensor, ends: torch.Tensor, rewards: torch.Tensor, discount: float):
    returns = [(rewards[start: end]*(torch.cumprod(torch.ones(end-start, device=rewards.device)*discount, dim=0))/discount).sum()
               for start, end in zip(starts, ends)]
    return torch.stack(returns)

@numba.jit(nopython=True, parallel=True)
def get_return(start, end, rewards):
    return rewards[start: end].sum()

@numba.jit(nopython=True, parallel=True)
def numba_sample_minibatch(observations, rewards, nb_train_trajectories, discount, nb_trajectories, pairs_per_trajectory):
    batched_source = []
    batched_target = []
    batched_returns = []
    batched_distance = []
    for _ in range(nb_trajectories):
        trajectory_idx = np.random.randint(nb_train_trajectories)
        observation = observations[trajectory_idx]
        reward = rewards[trajectory_idx]

        length = observation.shape[0]
        for _ in range(pairs_per_trajectory):
            start_idx = np.random.randint(length)
            end_idx = np.random.randint(start_idx, length)

            batched_source.append(observation[start_idx])
            batched_target.append(observation[end_idx])
            batched_returns.append(get_return(start_idx, end_idx, reward))
            batched_distance.append(discount**(end_idx-start_idx))

    return batched_source, batched_target, batched_returns, batched_distance


def sample_minibatch(observations, rewards, nb_train_trajectories, discount, nb_trajectories, pairs_per_trajectory):
    batched_source, batched_target, batched_returns, batched_distance = numba_sample_minibatch(observations, rewards, nb_train_trajectories, discount, nb_trajectories, pairs_per_trajectory)
    batched_source = np.stack(batched_source, axis=0)
    batched_target = np.stack(batched_target, axis=0)
    batched_returns = np.stack(batched_returns, axis=0)
    batched_distance = np.stack(batched_distance, axis=0)
    return dict(source=batched_source, target=batched_target, returns=batched_returns, distance=batched_distance)

class TunnelDataSet():
    def __init__(self, trajectory_data, test_size, discount):
        self.data = trajectory_data  # dictionary of lists of observations actions rewards and terminals
        self.discount = discount

        with torch.no_grad():
            self.test_data = dict(observations=[], actions=[], rewards=[], starts=[], ends=[], returns=[])
            current_test_size = 0
            test_index = 0
            while current_test_size < test_size:
                rewards = self.data["rewards"][test_index]
                starts, ends, returns = self.trajectory2pairs(rewards)
                self.test_data["observations"].append(self.data["observations"][test_index])
                self.test_data["actions"].append(self.data["actions"][test_index])
                self.test_data["rewards"].append(self.data["rewards"][test_index])
                self.test_data["starts"].append(starts+len(rewards))
                self.test_data["ends"].append(ends+len(rewards))
                self.test_data["returns"].append(returns)
                test_index += 1
                current_test_size += returns.shape[0]
            self.test_data = {k:torch.cat(v) for k, v in self.test_data.items()}
            test_source, test_target = self.batching_from_indices(self.test_data["observations"], self.test_data["starts"],
                                                     self.test_data["ends"])
            self.batched_test = dict(source=test_source, target=test_target, returns=self.test_data["returns"],
                                     distance=discount**(self.test_data["ends"]-self.test_data["starts"]))
            self.train_raw = {k: v[test_index:] for k, v in self.data.items()}
            self.nb_train_trajectories = len(self.train_raw["rewards"])

    @torch.no_grad()
    def sample_minibatch(self, nb_trajectories, pairs_per_trajectory):
        trajectory_idx = torch.randint(self.nb_train_trajectories, size=(nb_trajectories,),
                                       device=self.train_raw["rewards"][0].device)
        observations = [self.train_raw["observations"][i] for i in trajectory_idx]
        rewards = [self.train_raw["rewards"][i] for i in trajectory_idx]

        batched_source = []
        batched_target = []
        batched_returns = []
        batched_distance = []
        for observation, reward in zip(observations, rewards):
            length = observation.shape[0]
            rand_idx = torch.randint(length, size=(pairs_per_trajectory, 2), device=reward.device)
            start_idx = rand_idx.min(dim=1).values
            end_idx = rand_idx.max(dim=1).values

            batched_source.append(observation[start_idx])
            batched_target.append(observation[end_idx])
            batched_returns.append(get_returns(start_idx, end_idx, reward, self.discount))
            batched_distance.append(self.discount**(end_idx-start_idx))

        batched_source = torch.cat(batched_source, dim=0)
        batched_target = torch.cat(batched_target, dim=0)
        batched_returns = torch.cat(batched_returns, dim=0)
        batched_distance = torch.cat(batched_distance, dim=0)
        return dict(source=batched_source, target=batched_target, returns=batched_returns, distance=batched_distance)

    def batching_from_indices(self, observations, starts, ends):
        source = observations[starts]
        target = observations[ends]
        return source, target

    def trajectory2pairs(self, rewards):
        length = rewards.shape[0]
        start_indices = []
        end_indices = []
        for start in range(length):
            start_indices.append(torch.ones(length-start, dtype=torch.int64, device=rewards.device)*start)
            end_indices.append(torch.arange(start, length, dtype=torch.int64, device=rewards.device))
        starts, ends = torch.cat(start_indices), torch.cat(end_indices)
        returns = get_returns(starts, ends, rewards, self.discount)
        return starts, ends, returns


class TunnelModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, hidden_layers=1):
        super(TunnelModel, self).__init__()
        self.projection = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        hiddens = []
        for _ in range(hidden_layers):
            hiddens.append(nn.Linear(hidden_size*2, hidden_size*2))
            hiddens.append(nn.ReLU())
        self.hidden = nn.Sequential(*hiddens)
        self.return_head = nn.Linear(hidden_size*2, 1)
        self.distance_head = nn.Sequential(nn.Linear(hidden_size*2, 1), nn.Sigmoid())

    def forward(self, source, target):
        source_latent = self.projection(source)
        target_latent = self.projection(target)
        latent = torch.concat([source_latent, target_latent], dim=1)
        x = self.hidden(latent)
        returns = self.return_head(x)
        distance = self.distance_head(x)
        return returns[:,0], distance[:,0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="hopper-medium-v0")  # OpenAI gym environment name
    parser.add_argument("--id", default="T-1-1")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--discount", default=0.99, type=int)
    # parser.add_argument("--distance_head", default="discount", type=str)

    parser.add_argument("--batch_size", default=20480, type=int)
    parser.add_argument("--pairs", default=512, type=int)
    parser.add_argument("--iterations", default=1e6, type=int)
    parser.add_argument("--lr", default=0.0001, type=int)

    parser.add_argument("--log_period", default=1000, type=int)
    parser.add_argument("--test_size", default=100000, type=int)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = parser.parse_args()
    env = gym.make(args.env)

    results_dir = os.path.expandvars(os.path.expanduser(os.path.join('~/locallogs/tunnel', args.id)))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw_dataset = dict_to_tensor(env.get_dataset(), device=device)
    dataset = TunnelDataSet(divide2trajectories(raw_dataset), args.test_size, args.discount)
    model = TunnelModel(raw_dataset["observations"].shape[-1])
    model.to(device=device)

    loss_f = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    record = dict(step=[], return_loss=[], distance_loss=[], test_return_error=[], test_distance_error=[])

    for step in range(int(args.iterations)):
        """
        minibatch = dict_to_tensor(sample_minibatch(dataset.train_raw["observations"],
                                                      dataset.train_raw["rewards"],
                                                      dataset.nb_train_trajectories,
                                                      dataset.discount,
                                                      args.batch_size//args.pairs, args.pairs), device=device)
        """
        minibatch = dataset.sample_minibatch(args.batch_size//args.pairs, args.pairs)
        returns_pred, distance_pred = model(minibatch["source"], minibatch["target"])
        return_loss = loss_f(returns_pred, minibatch["returns"])
        distance_loss = loss_f(distance_pred, minibatch["distance"])
        loss = return_loss+distance_loss
        loss.backward()
        optimizer.step()


        if step % args.log_period == 0:
            with torch.no_grad():
                test_data = dataset.batched_test
                returns_pred, distance_pred = model(test_data["source"], test_data["target"])
                return_error = torch.mean(torch.abs(returns_pred-test_data["returns"]))
                distance_error = torch.mean(torch.abs(distance_pred-test_data["distance"]))

            record["step"].append(step)
            record["return_loss"].append(return_loss.item())
            record["distance_loss"].append(distance_loss.item())
            record["test_return_error"].append(return_error.item())
            record["test_distance_error"].append(distance_error.item())
            pd.DataFrame(record).to_csv(f"{results_dir}/model_training_logs.csv")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, results_dir+"/model.ckpt")
            print(f"iteration {step}, return loss {return_loss.item()}, distance_loss {distance_loss.item()},"
                  f"test return error {return_error.item()}, test distance error {distance_error.item()}")