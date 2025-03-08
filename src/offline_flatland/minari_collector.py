import argparse
import minari
import environment
import gymnasium as gym
from minari import DataCollector

dataset_id = "rail_env/flatland-v0.1"

def start_collecting(num_steps: int, env: DataCollector):
    for _ in range(num_steps):
        env.reset()
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='FlatlandDatasetCollector',
                    description='This is the dataset collection program for offline RL from flatland environment using Minari library.')
    parser.add_argument('-n', '--stepsnum', type=int, help='Number of environment restarts for the dataset', default=10)
    parser.add_argument('--width', type=int, help='Width for RailEnv field', default=30)
    parser.add_argument('--height', type=int, help='Height for RailEnv field', default=30)
    parser.add_argument('--agents', type=int, help='Count of agents for RailEnv', default=2)
    args = parser.parse_args()
    
    env = DataCollector(gym.make(dataset_id, width=args.width, height=args.height, number_of_agents=args.agents))
    start_collecting(args.stepsnum, env)
    
    # delete the test dataset if it already exists
    local_datasets = minari.list_local_datasets()
    if dataset_id in local_datasets:
        minari.delete_dataset(dataset_id)

    dataset = env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name="random_policy",
    )