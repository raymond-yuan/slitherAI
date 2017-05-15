import gym
import universe  # register the universe environments
import logging
from universe import wrappers
import numpy as np
import random
from universe import spaces

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# # env = gym.make('internet.SlitherIOEasy-v0')
# env = gym.make('internet.SlitherIO-v0')
# env.configure(remotes=1)  # automatically creates a local docker container
# env = wrappers.experimental.SafeActionSpace(env)
#
#
# ### ACTIONS ###
# left = env.action_space[0]
# right = env.action_space[1]
# left_dash = env.action_space[2]
# right_dash = env.action_space[3]
#
# forward = [('KeyEvent', 'space', False) ,('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
# fd_dash = [('KeyEvent', 'space', True) ,('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
#
# # probably redundant
# action_space = [left, right, left_dash, right_dash]
#
# observation_n = env.reset()

def gen_action(space=False, left=False, right=False):
    return [spaces.KeyEvent.by_name('space', down=space),
                spaces.KeyEvent.by_name('left', down=left),
                spaces.KeyEvent.by_name('right', down=right)]

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

        for t in range(200000):

            env.render()
            action_n = [env.action_space.sample() for ob in observation_n]
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action_n)
            if t % 100 == 0:
                print("REWARD", done)

            # print
            # print("Observation %d, reward %d, done, %d".format(observation, reward, done))
            # if done:
            #     break
# init_random()

def initial_population(initial_games = 1000, goal_steps = 2000, score_requirement = 10):
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    try:
    # iterate through however many games we want:
        for g in range(initial_games):
            score = 0
            # moves specifically from this environment:
            game_memory = []
            # previous observation that we saw
            prev_observation = []
            # for each frame in g

            for t in range(goal_steps):
                # if t % 1000 == 0:
                # env.render()
                if t % 200 == 0 or score > 0:
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
            # for t in range(2000):
            #     # if t % 1000 == 0:
            #     print t
            #     print("SCORE: ", t , score)
            #     action_n = [env.action_space.sample() for ob in observation_n]
            #     # do it!
            #     observation, reward, done, info = env.step(action_n)
            #
            # # env = gym.make('internet.SlitherIOEasy-v0')
# notice that the observation is returned FROM the action
            #     # so we'll store the previous observation here, pairing
            #     # the prev observation to the action we'll take.
            #     if len(prev_observation) > 0 :
            #         game_memory.append([prev_observation, action])
            #         prev_observation = observation
            #         score += reward[0]
            #     if done:
            #         break






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
    except:
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



#reinforcement learning step
def determine_turn(turn, observation_n, j, total_sum, prev_total_sum, reward_n):
    #for every 15 iterations, sum the total observations, and take the average
    #if lower than 0, change the direction
    #if we go 15+ iterations and get a reward each step, we're doing something right
    #thats when we turn
    if(j >= 10):
        if(total_sum/ float(j) ) <= 0.5:
            turn = True
        else:
            turn = False

        #reset vars
        total_sum = 0
        j = 0
        prev_total_sum = total_sum
        total_sum = 0

    else:
        turn = False
    if(observation_n != None):
        #increment counter and reward sum
        j+=1
        total_sum += reward_n

    return(turn, j, total_sum, prev_total_sum)

def determine_dash(dash, observation_n, k, t_sum, p_tsum, reward_n):
    if (k >= 10):
        if (t_sum / float(k)) <= 0.5:
            dash = True
        else:
            dash = False

        t_sum = 0
        k = 0
        p_tsum = t_sum
        t_sum = 0
    else:
        dash = False
    if(observation_n != None):
        #increment counter and reward sum
        k+=1
        t_sum += reward_n
    return dash, k, t_sum, p_tsum


def reinforce_run(training_data = []):

    #init variables
    env = gym.make('internet.SlitherIO-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    env = wrappers.experimental.SafeActionSpace(env)


    ### ACTIONS ###
        # left = env.action_space[0]
        # right = env.action_space[1]
        # left_dash = env.action_space[2]
        # right_dash = env.action_space[3]
        # forward = [('KeyEvent', 'Space', False) ,('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
        # fd_dash = [('KeyEvent', 'Space', True) ,('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]
    left = gen_action(left = True)
    right = gen_action(right = True)
    left_dash = gen_action(left = True, space = True)
    right_dash = gen_action(right = True, space = True)

    forward = gen_action()
    fd_dash = gen_action(space=True)

    action_space = [left, right, left_dash, right_dash, forward, fd_dash]

    observation_n = env.reset()


    # num of game iterations
    n = 0
    j = 0
    k = 0
    #sum of observations
    total_sum = 0
    prev_total_sum = 0

    # dash mechanics
    t_sum = 0
    p_tsum = 0
    turn = False
    dash = False
    action_n = [forward for ob in observation_n]
    prev_observation = []
    accepted_scores = []
    game_memory = []
    score = 0
    #main logic
    while True:
        #increment a counter for number of iterations
        n += 1

        #if at least one iteration is made, check if turn is needed
        if(n > 1):
            prev_observation = observation_n[0]
            game_memory.append([prev_observation, action_n])


            #if at least one iteration, check if a turn
            if(observation_n[0] != None):
                #store the reward in the previous score
                prev_score = reward_n[0]

                #should we turn?
                if(turn):
                    if (dash):
                        event = random.choice([left_dash, right_dash])
                        # event = [left_dash, right_dash][n % 2]
                        dash = False
                    else:
                        #pick a random event
                        #where to turn?
                        event = random.choice([left,right])
                        # event = [left, right][n % 2]
                    #perform an action
                    action_n = [event for ob in observation_n]
                    #set turn to false
                    turn = False

        elif (~turn):

            print "NO TURN"
            #if no turn is needed, go straight
            if (dash):
                action_n = [fd_dash for ob in observation_n]
                dash = False
            else:
                action_n = [forward for ob in observation_n]


        #if there is an obseravtion, game has started, check if turn needed
        if(observation_n[0] != None):
            turn, j, total_sum, prev_total_sum = determine_turn(turn, observation_n[0], j, total_sum, prev_total_sum, reward_n[0])
            dash, k, t_sum, p_tsum = determine_dash(dash, observation_n[0], k, t_sum, p_tsum, reward_n[0])
            score += reward_n[0]
            # turn, j, total_sum, prev_total_sum = determine_turn(turn, observation_n[0], j, total_sum, prev_total_sum, reward_n[0])
            # dash, k, total_sum, prev_total_sum = determine_dash(dash, observation_n[0], k, total_sum, prev_total_sum, reward_n[0])

        #save new variables for each iteration
        observation_n, reward_n, done_n, info = env.step(action_n)

        if done_n[0]:
            print "FINISHED GAME ", score
            if score >= 85:
                accepted_scores.append(score)

                # some stats here, to further illustrate the neural network magic!
                print('AVERAGE accepted score:', np.mean(accepted_scores))
                print('MEDIAN score for accepted scores:', np.median(accepted_scores))
                print(len(accepted_scores))
                for i in range(len(game_memory)):

                    # generate binary output for nn corresponding to action taken; [left, right, leftdash, right_dash]
                    output = gen_bin_output(len(action_space), action_space.index(data[1][0]))
                    game_memory[i][1] = output
                    # saving our training data [prev_observation, action]
                    # training_data.append([data[0], output])
                training_data.append(game_memory)

            game_memory = []
            prev_observation = []
            score = 0


        # env.render()
# reinforce_run()


try:
    training_data = []
    reinforce_run(training_data)
except KeyboardInterrupt:
    # just in case you wanted to reference later
    print "SAVING DATA"
    training_data_save = np.array(training_data)
    np.save('reinforcetraindata2.npy',training_data_save)
except:
    # just in case you wanted to reference later
    print "SOMETHING TERRIBLE BROKE: SAVING DATA"
    training_data_save = np.array(training_data)
    np.save('reinforcetraindata_esave.npy',training_data_save)
