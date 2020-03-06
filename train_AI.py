### This script contains the functions related to the shared AI.
### When run, this script will train the shared AI for a designated number of generations.

# import required libraries
from utilities import sample_weights, categorize_tokens, expand_inventory, expand_vector
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import regularizers
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model
from scipy.stats import uniform, randint, bernoulli


### This section contains the functions that run the game.

## function setup_tokens
# returns a linup of tokens shuffled to setup the game
def setup_tokens():
    # shuffle the four groups of tokens
    level_1 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    np.random.shuffle(level_1)
    level_2 = np.array([4, 4, 5, 5, 6, 6, 7, 7])
    np.random.shuffle(level_2)
    level_3 = np.array([8, 8, 9, 9, 10, 10, 11, 11])
    np.random.shuffle(level_3)
    level_4 = np.array([12, 12, 13, 13, 14, 14, 15, 15])
    np.random.shuffle(level_4)
    
    # stack groups into a single array and shuffle each group
    token_line = np.concatenate((level_1, level_2, level_3, level_4), axis = 0)
    
    # return the shuffled token lineup as a vector
    return token_line
	
## function die_roll
# takes an integer loot_count
# returns a number of spaces to move the player
def die_roll(loot_count):
    # roll the dice
    roll = randint.rvs(1, 4) + randint.rvs(1, 4)
    
    # reduce the roll by the amount of loot caried, but not below zero 
    roll = roll - loot_count
    if roll < 0:
        roll = 0
    
    # return the modified roll
    return roll
	
### A class to hold the current game state
class GameState:
    # initiate the game state by passing the number of players
    def __init__(self, players = 6):
        # players as defined on initation
        assert players >= 2 and players <= 6, "Must be between 2 and 6 players."
        self.players = players
        
        # shuffle the initial tokens
        self.tokens = setup_tokens()
        
        # set initial air supply to 25
        self.air_supply = 25
        
        # give each player an inventory of zero tokens (list of arrays)
        self.inventory = [np.array([]) for i in range(players)]
        
        # put each player on the sub (position -1) with downward heading (+1)
        self.positions = np.tile([-1, 1], (players, 1))
        
        # set the active player to player 0
        self.active_player = 0
        
        # set each player's score to 0
        self.player_scores = np.zeros(players)
        
        # track the game round
        self.round = 1
        
    # bound function to update the air supply
    def update_air_supply(self):
        # remove air from the supply equal to the active player's token count
        self.air_supply -= len(self.inventory[self.active_player])
        
    # bound function to collect a token
    def collect_token(self):
        # add a token to the active player's inventory
        self.inventory[self.active_player] = np.append(self.inventory[self.active_player], self.tokens[self.positions[self.active_player][0]])
        
        # remove the token from the lineup (set to -1)
        self.tokens[self.positions[self.active_player][0]] = -1
        
    # bound function to drop a token
    def drop_token(self, inventory_mask):
        # find the index of the selected token
        index_to_drop = np.where(inventory_mask == 1)[0][0]
        
        # place the token back in line
        self.tokens[self.positions[self.active_player][0]] = self.inventory[self.active_player][index_to_drop]
                                                                      
        # remove it from the player's inventory
        self.inventory[self.active_player] = np.delete(self.inventory[self.active_player], [index_to_drop])
        
    # bound function to start the next round
    def next_round(self, noise):
        # reset air supply to 25
        self.air_supply = 25
        
        # remove blanks from token row
        self.tokens = np.delete(self.tokens, np.where(self.tokens == -1)[0])
        
        # clean up tokens collected in the previous round
        for i in range(self.players):
            # score any tokens that players got back to the sub with
            if self.positions[i, 0] == -1:
                self.player_scores[i] += list(map(sum, self.inventory))[i]
            
            # drop any tokens that did not make it back
            else:
                # set up a counter to track how many tokens have been added to the current stack
                stacked = 0
            
                # loop through players
                for j in range(self.players):
                    # request drops until the player's inventory is empty
                    while len(self.inventory[j]) > 0:
                        # request which token to drop
                        selected_token = death_drop(categorize_tokens(self.inventory[j]), self.pass_gamestate(), noise)

                        # translate the mask back to an indx
                        selected_token = np.where(selected_token == 1)[0][0]

                        # add the selected tokens to the token row
                        # if this is the first token in a stack
                        if stacked == 0:
                            # add it to the end of the row
                            self.tokens = np.append(self.tokens, self.inventory[j][selected_token])
                            
                        # if the stack is not full, add to it
                        else:
                            self.tokens[-1] += self.inventory[j][selected_token]

                        # remove the tokens from the player's inventory
                        self.inventory[j] = np.delete(self.inventory[j], selected_token)
                        
                        # update the stack counter
                        stacked += 1
                        
                        # if the stack is full, reset the counter to 0
                        if stacked == 3:
                            stacked = 0
                        
        # fully clear player inventories
        self.inventory = [np.array([]) for i in range(self.players)]
        
        # set the active player to the deepest player
        self.active_player = np.argmax(self.positions[:, 0])
        
        # put each player on the sub (position -1) with downward heading (+1)
        self.positions = np.tile([-1, 1], (self.players, 1))
        
        # update the round counter
        self.round += 1
    
    # bound function that summarizes the gamestate in the vector format necessary to produce predictions
    def pass_gamestate(self):
        Round = self.round
        Tokens_Inventories = np.concatenate((expand_inventory(self.inventory, self.players)[self.active_player:, :], expand_inventory(self.inventory, self.players)[:self.active_player, :])).flatten()
        Air_Supply = self.air_supply
        Token_Positions = expand_vector(categorize_tokens(self.tokens))
        Player_Positions = np.concatenate((self.positions[self.active_player:, :], self.positions[:self.active_player, :])).flatten()
        Intermediate_Scores = np.concatenate((self.player_scores[self.active_player:], self.player_scores[:self.active_player])).flatten()
        
        return np.hstack((Round, Tokens_Inventories, Air_Supply, Token_Positions, Player_Positions, Intermediate_Scores))
		
## function take_turn
# takes a GameState plus game decision models and advances it by a turn
def take_turn(game, model_turn_around, model_pick_up, model_drop_token, noise):
    # reporter variables
    turn_taken = False
    turn_around = False
    pick_up = False
    drop = np.array([])
    
    # check that the active player is not already back on the submarine
    if not (game.positions[game.active_player, 0] == -1 and game.positions[game.active_player, 1] == -1):
        # note that the turn was not taken
        turn_taken = True
    
        # update the air supply
        game.update_air_supply()

        # determine whether the active player turns around
        if game.positions[game.active_player, 1] == 1 and game.positions[game.active_player, 0] != -1:
            if turn_around_decision(game.pass_gamestate(), model_turn_around, noise):
                # turn the player back toward the sub if True
                game.positions[game.active_player, 1] = -1;
                turn_around = True

        # roll the dice
        roll = die_roll(len(game.inventory[game.active_player]))

        # move the player along the token lineup until they consume their moves or reach the end
        while roll > 0:
            # check to see if the player is on the last available space when moving down
            if game.positions[game.active_player, 1] == 1:
                if game.positions[game.active_player, 0] == max(np.setdiff1d(np.arange(len(game.tokens)), game.positions[np.arange(len(game.positions)) != game.active_player][:, 0])):
                    break
            # check to see if the player is at the sub when moving up
            else:
                if game.positions[game.active_player, 0] == -1:
                    break
            
            # move the player a step in the appropriate direction
            game.positions[game.active_player, 0] += game.positions[game.active_player, 1]
            
            # continue moving without consuming the roll if the token is already occupied
            if game.positions[game.active_player, 0] in game.positions[np.arange(len(game.positions)) != game.active_player][:, 0]:
                continue
            
            # check to see if they have reached the sub
            if game.positions[game.active_player, 0] == -1:
                break
            
            # consume one movement from the roll
            roll = roll - 1

        # determine whether the active player picks up or drops a token
        # check to see if a token is available
        if game.tokens[game.positions[game.active_player, 0]] > -1:
            if pick_up_decision(game.pass_gamestate(), model_pick_up, noise):
                # pick up the token if True
                game.collect_token()
                pick_up = True
        elif len(game.inventory[game.active_player]) > 0:
            # select tokens to drop
            drop = drop_decision(categorize_tokens(game.inventory[game.active_player]), game.pass_gamestate(), model_drop_token, noise)
            
            # drop a token if any indicated
            if sum(drop > 0):
                game.drop_token(drop)
    
    # update the active player
    if game.active_player < game.players - 1:
        game.active_player += 1
    else:
        game.active_player = 0
        
    # check to see if oxygen has run out
    if game.air_supply <= 0 or np.all(game.positions[:, 0] == -1):
        # start the next round
        game.next_round(noise)
    
    # return a boolean indicating whether the game is over alongside descriptions of player's actions
    return(game.round <= 3, turn_taken, turn_around, pick_up, drop)


### This section contains functions necessary for training the AI.

## function instantiate model
# takes a tuple input shape, and optionally a dropout_rate and l2 lambda value
# returns a compiled model
def instantiate_model(input_shape, dropout_rate = 0, l2_lambda = 0):
    ## define model architecture
    # input
    input_layer = layers.Input(shape = input_shape)

    # dense
    first_dense_layer = layers.Dense(
        units = 256,
        activation = "relu",
        kernel_regularizer = regularizers.l2(l2_lambda)
    )(input_layer)

    # dropout
    first_dropout_layer = layers.Dropout(
        rate = dropout_rate
    )(first_dense_layer)

    # dense
    second_dense_layer = layers.Dense(
        units = 128,
        activation = "relu",
        kernel_regularizer = regularizers.l2(l2_lambda)
    )(first_dropout_layer)

    # dropout
    second_dropout_layer = layers.Dropout(
        rate = dropout_rate
    )(second_dense_layer)

    # softmax
    softmax_layer = layers.Dense(
        #units = np.unique(test[0][1]).shape[0],
        units = 2,
        activation = "softmax"
    )(second_dropout_layer)

    # input/output mapping
    model = keras.Model(
        inputs = input_layer,
        outputs = softmax_layer,
    )
    
    ## compile the model
    model.compile(
        loss = "binary_crossentropy",
        optimizer = optimizers.RMSprop(),
        metrics = ["accuracy"]
    )
    
    return model

## function simulate_game
# takes an integer number of players plus models for game decisions
# returns a dataframe of board states and a dataframe of player results
def simulate_game(players, model_turn_around, model_pick_up, model_drop_token, noise):

    # declare lists to store individual components of the game state
    TurnID = [] # int, indexing identifier
    Active_Player = [] # int, indexing identifier
    Round = [] # int
    Tokens_Inventories = [] # array with dim [players, 32], row 0 is inventory of active player
    Air_Supply = [] # int
    Token_Positions = [] # vector
    Player_Positions = [] # array with dim [players, 2], row 0 is active player
    Intermediate_Scores = [] # vector, index 0 is active player
    
    # additional lists to document the decisions made by the player on a given turn
    Turn_Around = [0] # bool, happens prior to board update
    Pick_Up = [0] # bool, happens after board update
    Drop = [[0 for i in range(32)]] # vector, happens after board update
    
    # initiate a new game state
    game = GameState(players)
    
    # start turn counter, document initial game state
    turn = 0
    TurnID.append(turn)
    Active_Player.append(game.active_player)
    Round.append(game.round)
    Tokens_Inventories.append(np.concatenate((expand_inventory(game.inventory, game.players)[game.active_player:, :], expand_inventory(game.inventory, game.players)[:game.active_player, :])).flatten())
    Air_Supply.append(game.air_supply)
    Token_Positions.append(expand_vector(categorize_tokens(game.tokens)))
    Player_Positions.append(np.concatenate((game.positions[game.active_player:, :], game.positions[:game.active_player, :])).flatten())
    Intermediate_Scores.append(np.concatenate((game.player_scores[game.active_player:], game.player_scores[:game.active_player])).flatten())
    
    # declare a flag for game end
    continue_game = True
    
    # take turns until the game is over
    while continue_game:
        # take a turn, collect the player actions taken
        continue_game, turn_taken, turn_around, pick_up, drop = take_turn(game, model_turn_around, model_pick_up, model_drop_token, noise)
        
        # skip to next turn if active player is already back at the sub
        if turn_taken == False:
            continue
        
        # document the game state
        turn += 1
        TurnID.append(turn)
        Active_Player.append(game.active_player)
        Round.append(game.round)
        Tokens_Inventories.append(np.concatenate((expand_inventory(game.inventory, game.players)[game.active_player:, :], expand_inventory(game.inventory, game.players)[:game.active_player, :])).flatten())
        Air_Supply.append(game.air_supply)
        Token_Positions.append(expand_vector(categorize_tokens(game.tokens)))
        Player_Positions.append(np.concatenate((game.positions[game.active_player:, :], game.positions[:game.active_player, :])).flatten())
        Intermediate_Scores.append(np.concatenate((game.player_scores[game.active_player:], game.player_scores[:game.active_player])).flatten())
        
        # document the player decisions
        Turn_Around.append(int(turn_around))
        Pick_Up.append(int(pick_up))
        Drop.append(expand_vector(drop))
    
    # wrap the board states into a dataframe
    Game_Log = pd.DataFrame({"TurnID": TurnID, "Active_Player": Active_Player, "Round": Round, "Tokens_Inventories": Tokens_Inventories, "Air_Supply": Air_Supply, "Token_Positions": Token_Positions, "Player_Positions": Player_Positions, "Intermediate_Scores": Intermediate_Scores, "Turn_Around": Turn_Around, "Pick_Up": Pick_Up, "Drop": Drop})
    
    # document the winner
    # ignoring tie breakers, all players lose when tied
    Placement = dict([(i, 0) for i in range(len(game.player_scores))])
    if not np.all(game.player_scores == game.player_scores[0]):
        Placement[np.argmax(game.player_scores)] = 1
    
    return Game_Log, Placement
    
## function build_training_set
# takes an integer number of games to simulate
# returns a list of couples containing the design matrix and target matrix for each decision function to train
def build_training_set(games, model_turn_around, model_pick_up, model_drop_token, noise):
    for i in range(games):
        # randomly select a player count from 2 to 6
        #players = np.random.choice(np.arange(2, 7))
        players = 6
        
        # simulate a game; remove the "4th round" row
        board_state, scores = simulate_game(players, model_turn_around, model_pick_up, model_drop_token, noise)
        board_state = board_state.drop(board_state.shape[0] - 1)
        
        # pick up and drop decisions occur after player movement and air supply are updated
        # shift columns around to reflect this
        board_state_tokens = board_state.copy()
        board_state_tokens.Air_Supply = board_state_tokens.Air_Supply.shift(-1)
        board_state_tokens.Player_Positions = board_state_tokens.Player_Positions.shift(-1)
        board_state_tokens.Pick_Up = board_state_tokens.Pick_Up.shift(-1)
        board_state_tokens.Drop = board_state_tokens.Drop.shift(-1)
        board_state_tokens = board_state_tokens.drop(board_state_tokens.shape[0] - 1)
             
        # initiate matrices if this is the first game
        if i == 0:
            # store the first row
            X_turn_around = np.hstack((board_state.Round[0], board_state.Tokens_Inventories[0], board_state.Air_Supply[0], board_state.Token_Positions[0], board_state.Player_Positions[0], board_state.Intermediate_Scores[0], board_state.Turn_Around[0]))
            X_pick_up = np.hstack((board_state_tokens.Round[0], board_state_tokens.Tokens_Inventories[0], board_state_tokens.Air_Supply[0], board_state_tokens.Token_Positions[0], board_state_tokens.Player_Positions[0], board_state_tokens.Intermediate_Scores[0], board_state_tokens.Pick_Up[0]))
            X_drop_token = np.hstack((board_state_tokens.Round[0], board_state_tokens.Tokens_Inventories[0], board_state_tokens.Air_Supply[0], board_state_tokens.Token_Positions[0], board_state_tokens.Player_Positions[0], board_state_tokens.Intermediate_Scores[0], board_state_tokens.Drop[0]))
            
            # remove the first row of the dataframe so it is not double-counted
            board_state = board_state.drop(0)
            
            # store the first result
            Y_turn_around = scores[0]
            Y_pick_up = scores[0]
            Y_drop_token = scores[0]
            
        # add rows from the dataframe to the turn around matrix and vector
        for j in range(board_state.shape[0]):
            # only add row if a turn around decision was possible
            if board_state.Player_Positions.iloc[j][1] != -1:
                X_turn_around = np.vstack((X_turn_around, np.hstack((board_state.Round.iloc[j], board_state.Tokens_Inventories.iloc[j], board_state.Air_Supply.iloc[j], board_state.Token_Positions.iloc[j], board_state.Player_Positions.iloc[j], board_state.Intermediate_Scores.iloc[j], board_state.Turn_Around.iloc[j]))))
                Y_turn_around = np.append(Y_turn_around, scores[board_state.Active_Player.iloc[j]])
        
        # add rows from the shifted dataframe to the other matrices and vectors
        for j in range(board_state_tokens.shape[0]):
            # determine whether a token was available to be picked up
            if board_state_tokens.Token_Positions[j][board_state_tokens.Player_Positions[j][0]] != 0:
                X_pick_up = np.vstack((X_pick_up, np.hstack((board_state_tokens.Round.iloc[j], board_state_tokens.Tokens_Inventories.iloc[j], board_state_tokens.Air_Supply.iloc[j], board_state_tokens.Token_Positions.iloc[j], board_state_tokens.Player_Positions.iloc[j], board_state_tokens.Intermediate_Scores.iloc[j], board_state_tokens.Pick_Up.iloc[j]))))
                Y_pick_up = np.append(Y_pick_up, scores[board_state_tokens.Active_Player.iloc[j]])
            
            # otherwise add to the drop data instead
            else:
                X_drop_token = np.vstack((X_drop_token, np.hstack((board_state_tokens.Round.iloc[j], board_state_tokens.Tokens_Inventories.iloc[j], board_state_tokens.Air_Supply.iloc[j], board_state_tokens.Token_Positions.iloc[j], board_state_tokens.Player_Positions.iloc[j], board_state_tokens.Intermediate_Scores.iloc[j], board_state_tokens.Drop.iloc[j]))))            
                Y_drop_token = np.append(Y_drop_token, scores[board_state_tokens.Active_Player.iloc[j]])
            
    return [(X_turn_around, Y_turn_around), (X_pick_up, Y_pick_up), (X_drop_token, Y_drop_token)]
    
## function train_model
# takes a model object, a design matrix, a target vector, target weights, and an optional verbose flag
# trains model until validation error stops decreasing, then returns the model object
def train_model(model, X, Y, weights, verbose = 0):
    # declare an early stopping callback
    ES_callback = EarlyStopping(patience = 5, restore_best_weights = True)
    
    # fit the model
    model.fit(
        x = X,
        y = to_categorical(Y),
        batch_size = 32,
        epochs = 100,
        verbose = verbose,
        validation_split = 0.2,
        callbacks = [ES_callback]
    )
    
    # return the trained model plus the number of epochs run
    return (model, ES_callback.stopped_epoch + 1)

    
## function compare_models
# takes two tuples of models and an integer number of games
# simulates games and returns the number of wins for each
def compare_models(model_set_1, model_set_2, games = 1000, noise = 0.1):
    # initiate counter to track win totals
    wins = np.zeros(2)
    
    for i in range(games):
        # initiate a new game state
        game = GameState(6)

        # start turn counter; declare a flag for game end
        turn = 0
        continue_game = True

        # randomly assign model_set_1 and model_set_2 to even/odd turns
        model_set_1_turns = randint.rvs(0, 1)

        # take turns until the game is over
        while continue_game:
            # use model_set_1 on half the turns
            if turn % 2 == model_set_1_turns:
                # take a turn
                continue_game, turn_taken, turn_around, pick_up, drop = take_turn(game, *model_set_1, noise)

            # use model_set_2 on the other half
            else:
                # take a turn
                continue_game, turn_taken, turn_around, pick_up, drop = take_turn(game, *model_set_2, noise)

            # increment turn counter
            turn += 1

        # document the winner    
        if np.argmax(game.player_scores) % 2 == model_set_1_turns:
            wins[0] += 1
        else:
            wins[1] += 1
    
    # return the win totals
    return wins
    

### This section contains functions to make decisions during gameplay.
    
## function turn_around_decision
# takes a model and a 
# returns a boolean indicating whether to turn around or not
def turn_around_decision(gamestate, model_turn_around, noise):
    # add "pickup true" to gamestate
    gamestate = np.reshape(np.hstack((gamestate, 1)), (1, 245))
    
    # predict probability of winning if the player turns around and add noise
    prob = (1 - noise) * model_turn_around.predict(gamestate)[0][1] + noise * uniform.rvs()
    
    # Bernoulli trial with probability
    return bernoulli.rvs(prob)
	
## function pick_up_decision
# returns a boolean indicating whether to pick up the token or not
def pick_up_decision(gamestate, model_pick_up, noise):
    # add "drop true" to gamestate
    gamestate = np.reshape(np.hstack((gamestate, 1)), (1, 245))
    
    # predict probability of winning if the player picks up the token and add noise
    prob = (1 - noise) * model_pick_up.predict(gamestate)[0][1] + noise * uniform.rvs()

    # Bernoulli trial with probability
    return bernoulli.rvs(prob)
	
## function drop_decision
# returns an index indicating which token to drop
def drop_decision(inventory, gamestate, model_drop_token, noise):
    # create a vector of zeroes 
    inventory_mask = np.zeros(len(inventory))
    
    # declare a vector to hold predictions
    drop_probs = np.array([])
    
    # extend gamestate with a vector of zeroes
    gamestate_drop_options = np.reshape(np.hstack((gamestate, np.zeros(32))), (1, 276))
    
    # predict probability of winning if no tokens are dropped
    drop_probs = np.append(drop_probs, (1 - noise) * model_drop_token.predict(gamestate_drop_options)[0][1] + noise * uniform.rvs())
    
    # predict probability of winning if each token in inventory is dropped
    for i in range(len(inventory)):
        # create a vector representation of dropping each token
        drop_options = np.zeros(32)
        drop_options[i] = 1
        
        # extend gamestate with the drop options vector
        gamestate_drop_options = np.reshape(np.hstack((gamestate, drop_options)), (1, 276))
        
        # make prediction
        drop_probs = np.append(drop_probs, (1 - noise) * model_drop_token.predict(gamestate_drop_options)[0][1] + noise * uniform.rvs())
    
    # drop the highest probability token
    if np.argmax(drop_probs) != 0:
        inventory_mask[np.argmax(drop_probs) - 1] = 1
        
    return inventory_mask
    
## function death_drop
# returns an index indicating which token to drop
def death_drop(inventory, gamestate, noise):
    # create a vector of zeroes
    inventory_mask = np.zeros(len(inventory))

    # select a token at random
    inventory_mask[randint.rvs(0, len(inventory))] = 1

    return inventory_mask
    

## function training_iteration
# takes a number of generations to train, a number of games per generation, and a tuple of models
# compares models at the end of training to a baseline set
# returns trained models if they outperform by at least 55% to 45%
# otherwise, returns baseline models
def training_iteration(generations, baseline_models, games_per_generation = 100):
    # instantiate models for each of the game decisions
    model_turn_around = instantiate_model((245, ), 0.3, 0)
    model_pick_up = instantiate_model((245, ), 0.3, 0)
    model_drop_token = instantiate_model((276, ), 0.3, 0)

    # simulate game data, use it to update models, and repeat
    for i in range(generations):
        # set the current noise level for game decisions
        noise = 0.5 - 0.4 * (i / generations)
        
        # simulate games using baseline models
        simulated_game_data = build_training_set(games_per_generation, *baseline_models, noise)
        
        # train the models on the current set of simulated games
        model_turn_around, epoch_turn_around = train_model(model_turn_around, simulated_game_data[0][0], simulated_game_data[0][1], sample_weights(simulated_game_data[0][1], 6))
        model_pick_up, epoch_pick_up = train_model(model_pick_up, simulated_game_data[1][0], simulated_game_data[1][1], sample_weights(simulated_game_data[1][1], 6))
        model_drop_token, epoch_drop_token = train_model(model_drop_token, simulated_game_data[2][0], simulated_game_data[2][1], sample_weights(simulated_game_data[2][1], 6))
        
        # print the number of epochs each model was trained prior to stopping
        print("Generation: %i" % i)
        print(epoch_turn_around, epoch_pick_up, epoch_drop_token)
    
    # compare the trained models to the baseline models
    results = compare_models((model_turn_around, model_pick_up, model_drop_token), baseline_models)
    
    # return the trained models if they exceed previous performance by a 55% to 45% margin
    if results[0]/np.sum(results) >= 0.55:
        print("Improvement")
        return (model_turn_around, model_pick_up, model_drop_token)
    # otherwise return the baseline models
    else:
        print("No Improvement")
        return baseline_models
    

## executed code
# continue until manually stopped
#while True:
    # load the current baseline models
#    baseline_models = (load_model("model_turn_around.h5"), load_model("model_pick_up.h5"), load_model("model_drop_token.h5"))
    
    # train the models for 100 generations of 100 games
#    new_models = training_iteration(100, baseline_models)

    # save the returned models
#    new_models[0].save("model_turn_around.h5")
#    new_models[1].save("model_pick_up.h5")
#    new_models[2].save("model_drop_token.h5")
