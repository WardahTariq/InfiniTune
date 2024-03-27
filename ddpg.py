import fabric
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import copy
from datetime import datetime
import matplotlib.pyplot as plt



from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from InfiniTuneEnv import InfiniTuneEnv
from Parser import Parser
from keras.optimizers import Adam
import timeit




min_vals = np.array([2, '128MB', '4GB', '4MB'])
max_vals = np.array([8, '4GB', '6GB', '30MB'])
default_vals = np.array([2, '128MB', '4GB', '4MB'])
knob_names = ['max_parallel_workers_per_gather', 'shared_buffers', 'effective_cache_size', 'work_mem']
knob_types = ['integer', 'size', 'size', 'size']
average_rewards = []
interval=10

with open('./tmpLog/TraininglogFile', 'w'):
    pass


def epsilon_greedy_policy(policy_action, epsilon):
    # Creating epsilon greedy probabilities to sample from.
    is_random = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
    if is_random:
        # [0,1) uniform random
        action = np.random.uniform()
    else:
        action = policy_action
    return action, is_random


def gaussian_noise_policy(policy_action, sigma):
    noise = sigma * np.random.randn(1)[0]
    action = policy_action + noise
    return action, noise


def get_range_raw(min_vals, max_vals, default_vals):
    min_raw_vals = []
    max_raw_vals = []
    default_raw_vals = []
    for i in range(len(min_vals)):
        min_raw_val = Parser().get_knob_raw(min_vals[i], knob_types[i])
        max_raw_val = Parser().get_knob_raw(max_vals[i], knob_types[i])
        default_raw_val = Parser().get_knob_raw(default_vals[i], knob_types[i])
        min_raw_vals.append(min_raw_val)
        max_raw_vals.append(max_raw_val)
        default_raw_vals.append(default_raw_val)
    return min_raw_vals, max_raw_vals, default_raw_vals

min_raw_vals, max_raw_vals, default_raw_vals = get_range_raw(min_vals, max_vals, default_vals)

print(default_raw_vals)
print(min_raw_vals)
print(max_raw_vals)


def get_config_knobs(knob_vals):
    config_knobs = []
    for i in range(len(knob_vals)):
        knob_val = Parser().get_knob_readable(knob_vals[i], knob_types[i])
        config_knobs.append(knob_val)
    return config_knobs


def load_weights(actor, critic, actor_file="actor.weights.h5",
                 critic_file="critic.weights.h5",target_actor_file='actor_target.weights.h5',target_critic_file='critic_target.weights.h5'):
    try:
        actor.model.load_weights(actor_file)
        critic.model.load_weights(critic_file)
        actor.target_model.load_weights(target_actor_file)
        critic.target_model.load_weights(target_critic_file)
        print("Weight load successfully")
    except:
        print("Cannot find the weight")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def write_log(f, current_state, readable_state, episode_i):
    f.write('{}: [{}, {}, {}, {}], [{}, {}, {}, {}]\n'.format(episode_i,
        current_state[0], current_state[1], current_state[2], current_state[3],
        readable_state[0], readable_state[1], readable_state[2], readable_state[3]))
    f.flush()





def ddpgTune():
    DEBUG = True    #Print intermediate results for debugging
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.9 # 1.0     #Discount Factor
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.005    #Learning rate for Actor  1/32 ~ 0.03  , 0.0001
    LRC = 0.005     #Lerning rate for Critic
    TRAIN_C = 20 # Training epochs for Critic
    TRAIN_A = 20 # Training epochs for Actor

    action_dim = 1  # set value (0,1)
    tuning_knobs_num = 4 # of tuning knobs
    # state = (tuning_knobs_value, current_knob_id)
    state_dim = tuning_knobs_num + 1 # of state
    
    seed = 1234
    set_seed(seed)

    episode_count = 50 #300 #400 #000
    step = 0

    # Exploration
    explore_episodes = 200
    explore_iters = tuning_knobs_num * explore_episodes

    # Exploration, epsilon greedy
    epsilon_begin = 0.5
    epsilon_end = 0.05

    # Exploration, gaussian noise
    sigma_begin = 0.3
    sigma_end = 0

    default_latency = 0.  # ms

    #Tensorflow GPU optimization

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    actor = ActorNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a InfiniTune environment
    env = InfiniTuneEnv(min_raw_vals, max_raw_vals, default_raw_vals, knob_names)

    # Now load the weight
    print("Now we load the weight")
    load_weights(actor, critic)

    f = open('./tmpLog/Training_w.log', 'w')
    latest_average_reward=[]

    
    print("InfiniTune experiment Start.")
    for episode_i in range(episode_count):

        print("Episode : " + str(episode_i) )

        initial_state = env.reset()
        current_state = initial_state
        episode_reward = 0
        buffs = []
        while True:

            if step > explore_iters:
                epsilon = epsilon_end
                sigma = sigma_end
            else:
                epsilon = epsilon_begin - (epsilon_begin - epsilon_end) * 1.0 * step / explore_iters
                print(f"epsilon {epsilon}")
            policy_action = actor.model.predict(np.array([current_state]), batch_size=1)[0][0]
            print(f"policy action {policy_action}")
            # epsilon greedy for exploration
            action, is_random = epsilon_greedy_policy(policy_action, epsilon)
            print(f"action {action} is_random {is_random}")

            # gaussian noise for exploration
            # action, is_random = gaussian_noise_policy(policy_action, sigma)

            # constraint [0, 1]
            action = min(action, 1.0)
            action = max(action, 0.0)

            print(f"current_state {current_state}")
            nextstate, reward, is_terminal, debug_info = env.step(action, current_state)
            print(f"nextstate {nextstate}")
            transition = [current_state, action, reward, nextstate, is_terminal]
            print(f"current state {current_state}")
            X_samples, X_currstates, X_nextstates, X_rewards, X_actions = buff.sort_batch(batch_size=BATCH_SIZE)
            print(f" sort_batch {X_samples} {X_currstates} {X_nextstates} {X_rewards} {X_actions}")
            if len(X_samples) > 0:
                print(f"X_nextstates {X_nextstates}")
                X_nextactions = actor.target_model.predict(np.array(X_nextstates), batch_size=len(X_nextstates))
                print(f"X_nextactions {X_nextactions}")
                Y_nextstates = critic.target_model.predict([np.array(X_nextstates), np.array(X_nextactions)], batch_size=len(X_nextstates))
                print(f"Y_nextstates {Y_nextstates}")
                Y_minibatch = np.zeros([len(X_samples), 1])
                print(f"Y_minibatch {Y_minibatch}")
                i = 0
                for x_state, x_action, x_reward, x_nextstate, x_is_terminal in X_samples:
                    if x_is_terminal:
                        y = x_reward
                    else:
                        y = x_reward + GAMMA * Y_nextstates[i]
                    Y_minibatch[i][0] = y
                    i += 1
                print(f"Y_minibatch {Y_minibatch}")    

                # update critic network
                critic.model.fit([np.array(X_currstates), np.array(X_actions)], np.array(Y_minibatch), batch_size=len(X_currstates), epochs=TRAIN_C, verbose=0)

                for i in range(TRAIN_A):
                    a_for_grad = actor.model.predict([np.array(X_currstates)], batch_size=len(X_currstates))
                    grads = critic.gradients(X_currstates, a_for_grad)
                    mean_grads = 1.0 / len(X_currstates) * grads
                    actor.train(X_currstates, mean_grads)
                    # if grads is not None:
                    #     mean_grads = 1.0 / len(X_currstates) * grads
                    #     actor.train(X_currstates, mean_grads)
                    # else:
                    #     print("Gradients computation failed, skipping actor training for this batch.")

                actor.target_train()
                critic.target_train()
            else: # initial default config
                
                tmp_current_state = copy.copy(initial_state)
                tmp_next_state = copy.copy(initial_state)
                tmp_current_state[tuning_knobs_num] = step 
                tmp_next_state[tuning_knobs_num] = step + 1  
                transition = [tmp_current_state, initial_state[step], 0, tmp_next_state, is_terminal]
                print(f"transition {transition}")
                nextstate = initial_state


            buffs.append(transition)
            print(f"buffs {buffs}")
            episode_reward += reward
            current_state = nextstate

            if DEBUG is True:
                print("Episode", episode_i, "Step", step,
                      "Action", action, "Random", is_random,
                      "Reward", reward, "Sample Size", len(X_samples))
            step += 1
            if is_terminal:
                break


        # print (current_state)
        # convert current knob to config knob
        rescaled_vals = Parser().rescaled(min_raw_vals, max_raw_vals, current_state[:tuning_knobs_num])
        config_vals = get_config_knobs(rescaled_vals)

        # print (config_vals)
        write_log(f, current_state, config_vals, episode_i)
        #f.write(current_state)
        env.change_conf(config_vals)

        # run tpch query to get reward
        config_reward = env.run_experiment()

        print(f'----------{config_reward}-----------------')

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open('./tmpLog/TraininglogFile', 'a') as log_file:
            log_file.write(f"{current_time}: {config_reward}\n")

        if episode_i == 0:
            default_reward = config_reward
            # buff.burn_in()

        # scaled reward, latency: 1 - new/default
        # As config_reward decreases scaled_reward increases.
        # Goal is to maximize scaled_reward,hence minimize config_reward(latency)
        scaled_reward = 100 * (1 - config_reward * 1.0 / default_reward)
        print('reward ', config_reward, 'scaled ', scaled_reward)
        for transition in buffs:
            transition[2] = scaled_reward
            buff.append(transition)
        latest_average_reward.append(config_reward)
        if episode_i % interval == 0 and episode_i != 0:
            average_reward = np.mean(latest_average_reward)
            average_rewards.append(average_reward)
            latest_average_reward=[]
            # print(f'Average reward for episodes {episode_i-10}-{episode_i}: {average_reward}')

        # save, udf and upload
        env.save_and_upload()
    f.close()
    print("Training finished.")
    x = np.arange(interval, episode_count, interval)
    plt.plot(x, average_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Latency)')
    plt.title('Learning Curve - Running Average of previous 100 rewards')
    plt.show()
   


    actor.save_model_weights('./tmp/actor.weights.h5','./tmp/actor_target.weights.h5')
    critic.save_model_weights('./tmp/critic.weights.h5','./tmp/critic_target.weights.h5')


if __name__ == "__main__":
    ddpgTune()

