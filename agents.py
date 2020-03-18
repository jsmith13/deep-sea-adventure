### This script defines the player agent class.

# import required libraries
import numpy as np
import pandas as pd
from collections import deque
import random
from keras import Model, layers, regularizers, optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import uniform, randint

## a class to hold the player experiences and training functions
class PlayerAgent:
    # inititate the agent
    def __init__(self, players):      
        # training parameters
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.batch_size = 512

        # a queue to hold trials and results
        self.turn_around_memory = deque(maxlen = 2000)
        self.pick_up_memory = deque(maxlen = 2000)
        self.drop_memory = deque(maxlen = 2000)
        
        # translate number of players into size of game state vector
        self.game_state_size = 35 * players + 34
        
        # decision and target models 
        # binary decision
        self.turn_around_model = self.create_model(2)
        self.turn_around_target_model = self.create_model(2)
        # binary decision
        self.pick_up_model = self.create_model(2)
        self.pick_up_target_model = self.create_model(2)
        # 32 possible tokens to drop and no drop
        self.drop_model = self.create_model(33)
        self.drop_target_model = self.create_model(33)
                
    # initiate a keras network
    def create_model(self, available_actions):
        ## define model architecture mapping a game state vector to a decision vector
        # input
        input_layer = layers.Input(shape = (self.game_state_size, ))
        
        # dense
        first_dense_layer = layers.Dense(
            units = 256,
            activation = "relu",
            kernel_regularizer = regularizers.l2(0.01)
        )(input_layer)
        
        # dense
        second_dense_layer = layers.Dense(
            units = 128,
            activation = "relu",
            kernel_regularizer = regularizers.l2(0.01)
        )(first_dense_layer)
        
        # output
        output_layer = layers.Dense(
            units = available_actions
        )(second_dense_layer)
        
        # compile the model
        model = Model(
            inputs = input_layer,
            outputs = output_layer
        )
        
        model.compile(
            loss = "mean_squared_error",
            optimizer = optimizers.RMSprop()
        )
        
        return model
        
    # make a turn around decision
    def turn_around_decision(self, gamestate):
        # decide if the model output or a random guess will be used
        if uniform.rvs(0, 1) <= self.epsilon:
            # randomly decide whether to turn around
            turn = np.eye(2)
            np.random.shuffle(turn)
            turn = np.argmax(turn[0, ])
        else:
            # generate a Q-table for the current gamestate
            turn = self.turn_around_model.predict(gamestate)
            
            # take the action with the highest Q-value
            turn = np.argmax(turn)
            
        # return the decision as a boolean
        return turn
        
    # make a pick up decision
    def pick_up_decision(self, gamestate):
        # decide if the model output or a random guess will be used
        if uniform.rvs(0, 1) <= self.epsilon:
            # randomly decide whether to pick up a token
            pickup = np.eye(2)
            np.random.shuffle(pickup)
            pickup = np.argmax(pickup[0, ])
        else:
            # generate a Q-table for the current gamestate
            pickup = self.pick_up_model.predict(gamestate)
            
            # take the action with the highest Q-value
            pickup = np.argmax(pickup)
            
        # return the decision as a boolean
        return pickup
        
    # make a drop decision
    def drop_decision(self, gamestate):
        # decide if the model output or a random guess will be used
        if uniform.rvs(0, 1) <= self.epsilon:
            # no drop if inventory is empty
            if sum(gamestate[1:33] != -1) == 0:
                drop = 0
            else:
                # randomly decide whether to drop a token from those available
                drop = randint.rvs(0, sum(gamestate[1:33] != -1) + 1)
        else:
            # generate a Q-table for the current gamestate
            selected_action = self.pick_up_model.predict(gamestate)
            
            # take the action with the highest Q-value
            drop = np.argmax(selected_token[0:(sum(gamestate[1:33] != -1) + 1)])
            drop = int(drop)
            
        # return the decision as an integer
        # 1-33 mean drop the corresponding item
        # 0 means no drop
        return drop
    
    # remove rows where no decision was mode from a game log, action log pair
    def truncate_game_log(self, state_log, action_log):
        # create a list of rows to remove
        to_remove = []
        
        # find rows where a decision was not made
        for row in range(state_log.shape[0]):
            if np.isnan(action_log[row, 0]):
                to_remove.append(row)
                
        # remove those rows from both logs
        state_log =  np.delete(state_log, to_remove, axis = 0)
        action_log = np.delete(action_log, to_remove, axis = 0)
        
        # return the logs
        return (state_log, action_log)
    
    # calculate reward from gamestate and action log
    def calculate_reward(self, state_log, action_log, active_player, placement):
        # number of players
        players = placement.shape[0]
        
        # declare a vector to hold the reward
        reward = np.zeros(shape = action_log.shape)
        
        # start score tracker at zero
        current_score = 0
        
        # find rows where points were scored
        for row in range(state_log.shape[0]):
            if state_log[row, -players] > current_score:
                # assign reward for newly scored points
                reward[row, 0] = state_log[row, -players] - current_score
                
                # update current score
                current_score = state_log[row, -players]
                
        # add additional points for winning the game
        if placement[active_player] == 1:
            reward[-1, 0] += 500
            
        # return the reward vector
        return reward
    
    # store a game data log in agent memory
    def store_game(self, game_log):
        # process turn around (0), pick up (1), then drop actions (2)
        for decision in range(3):
            # loop through players
            for player in range(len(game_log[0][0])):
                # remove rows where no decision was made
                truncated = self.truncate_game_log(game_log[decision][0][player], game_log[decision][1][player])
                
                # calculate reward
                reward = self.calculate_reward(truncated[0], truncated[1], player, game_log[3])
                
                # create a matrix of new states by shifting game states up one row
                new_state = truncated[0][1:]
                new_state = np.vstack((new_state, np.zeros(shape = (1,new_state.shape[1]))))
                
                # create a vector to mark game end
                end = np.zeros(shape = truncated[1].shape)
                end[-1] = 1
                
                # store in the memory queue
                # gamestate, decision, reward, new state, end
                if decision == 0:
                    for i in range(truncated[0].shape[0]):
                        self.turn_around_memory.append((truncated[0][i], truncated[1][i], reward[i], new_state[i], end[i]))
                elif decision == 1:
                    for i in range(truncated[0].shape[0]):
                        self.pick_up_memory.append((truncated[0][i], truncated[1][i], reward[i], new_state[i], end[i]))
                elif decision == 2:
                    for i in range(truncated[0].shape[0]):
                        self.drop_memory.append((truncated[0][i], truncated[1][i], reward[i], new_state[i], end[i]))
    
    # train a model
    def train_model(self):
        # process turn around (0), pick up (1), then drop decisions (2)
        for decision in range(3):
            # select the appropriate models and memory queue for the current decision
            if decision == 0:
                model = self.turn_around_model
                target_model = self.turn_around_target_model
                memory = self.turn_around_memory
            elif decision == 1:
                model = self.pick_up_model
                target_model = self.pick_up_target_model
                memory = self.pick_up_memory
            elif decision == 2:
                model = self.drop_model
                target_model = self.drop_target_model
                memory = self.drop_memory
        
            # abort if there are not enough training examples
            if len(memory) < self.batch_size:
                return
            
            # randomly sample (without replacement) from the memory queue
            samples = random.sample(memory, self.batch_size)
            
            # declare arrays to hold the gamestate and target sets
            gamestate_array = np.empty(shape = (0, model.input_shape[1]))
            target_array = np.empty(shape = (0, model.output_shape[1]))
            
            # process each sample and compile into training arrays
            for sample in samples:
                # unpack the tuple
                gamestate, decision, reward, new_state, end = sample
                
                # reshape gamestate and new_state into a single-row arrays
                gamestate = np.reshape(gamestate, (1, gamestate.shape[0]))
                new_state = np.reshape(new_state, (1, new_state.shape[0]))
                
                # generate a target Q-table from the target model
                target = target_model.predict(gamestate)
                
                # last action of the round, no future reward
                if end:
                    # update the target Q-table with results from this sample
                    target[0][int(decision)] = reward
                else:
                    # not the last action, add predicted future reward
                    Q_future = max(target_model.predict(new_state)[0])
                    target[0][int(decision)] = reward + Q_future * self.gamma
                
                # append the gamestate and target sets to the appropriate arrays
                gamestate_array = np.vstack((gamestate_array, gamestate))
                target_array = np.vstack((target_array, target))
                
            # add an early stopping callback
            ES_callback = EarlyStopping(patience = 5, restore_best_weights = True)
            
            # fit the decision model to the target Q-table
            model.fit(gamestate_array, target_array, batch_size = 64, epochs = 100, verbose = 0)
        
    # update the target weights
    def update_target_weights(self):
        # update turn around weights
        weights = self.turn_around_model.get_weights()
        target_weights = self.turn_around_target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.turn_around_target_model.set_weights(target_weights)
        
        # update pick up weights
        weights = self.pick_up_model.get_weights()
        target_weights = self.pick_up_target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.pick_up_target_model.set_weights(target_weights)
        
        # update turn around weights
        weights = self.drop_model.get_weights()
        target_weights = self.drop_target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.drop_target_model.set_weights(target_weights)
        
    # save models
    def save_models(self, filepath = "./models/", identifier = ""):
        self.turn_around_model.save(filepath + "turn_around_model_" + identifier)
        self.pick_up_model.save(filepath + "pick_up_model_" + identifier)
        self.drop_model.save(filepath + "drop_model_" + identifier)
    