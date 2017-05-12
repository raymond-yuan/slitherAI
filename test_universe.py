import gym
import universe  # register the universe environments
import logging
from universe import wrappers
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
env = gym.make('internet.SlitherIO-v0')
env.configure(remotes=1)  # automatically creates a local docker container
env = wrappers.experimental.SafeActionSpace(env)


### ACTIONS ###
left = env.action_space[0]
right = env.action_space[1]
left_dash = env.action_space[2]
right_dash = env.action_space[3]

# probably redundant
action_space = [left, right, left_dash, right_dash]

observation_n = env.reset()

# while True:
#   action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
#   observation_n, reward_n, done_n, info = env.step(action_n)
#   env.render()
def gen_bin_output(size, values):
    """
    This function will generate a binary output given a output size for the
    array and the indexes at which need to be one.

    Example usage:
    gen_bin_output(4, [0, 1]) -> [1, 1, 0, 0]
    gen_bin_output(4, 0) -> [1, 0, 0, 0]
    """
    out = np.zeros(size)
    out[values] = 1
    return out

def testing():
    for i in range(1000):
        # if i == 0:
        #     action_n = [env.action_space.sample() for ob in observation_n]
        # else:
        #     action_n = env.action_space.sample()
        action_n = [env.action_space.sample() for ob in observation_n]

        observation, reward, done, info = env.step(action_n)
        # print("ACTION TAKEN ", action_n)
        # print("HELLO???", reward)
# testing()

def init_random(num_games = 5):
    """
    Quick initial random agent
    """
    for ep in range(num_games):
        env.reset()

        for t in range(20000):
            env.render()
            action_n = [env.action_space.sample() for ob in observation_n]
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action_n)

            # print
            # print("Observation %d, reward %d, done, %d".format(observation, reward, done))
            if done:
                break
init_random()

def initial_population(initial_games = 1000, goal_steps = 2000, score_requirement = 50):
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []

    # iterate through however many games we want:
    for g in range(1):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in g
        t = 0
        while t < 2000:
            # if t % 1000 == 0:
            t += 1
            env.render()
            print("SCORE: ", t , score)
            action_n = [env.action_space.sample() for ob in observation_n]
            # do it!
            observation, reward, done, info = env.step(action_n)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
                prev_observation = observation
                score += reward[0]
            if done:
                break
        # for t in range(2000):
        #     # if t % 1000 == 0:
        #     print t
        #     print("SCORE: ", t , score)
        #     action_n = [env.action_space.sample() for ob in observation_n]
        #     # do it!
        #     observation, reward, done, info = env.step(action_n)
        #
        #     # notice that the observation is returned FROM the action
        #     # so we'll store the previous observation here, pairing
        #     # the prev observation to the action we'll take.
        #     if len(prev_observation) > 0 :
        #         game_memory.append([prev_observation, action])
        #         prev_observation = observation
        #         score += reward[0]
        #     if done:
        #         break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # generate binary output for nn corresponding to action taken; [left, right, leftdash, right_dash]
                output = gen_bin_output(len(action_space), action_space.index(data[1]))

                # saving our training data [prev_observation, action]
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('train.npy',training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', np.mean(accepted_scores))
    print('Median score for accepted scores:', np.median(accepted_scores))
    print(len(accepted_scores))

    return training_data

# print("GENERATING INITIAL POPULATIONS")
# initial_population()
