### This script contains the functions that maintain the game state.

# import required libraries
from utilities import categorize_tokens, expand_inventory, expand_vector, death_drop
import numpy as np
import pandas as pd
from scipy.stats import randint
	
## a class to hold the current game state
class GameState:
    # initiate the game state by passing the number of players
    def __init__(self, players = 6):
        # players as defined on initation
        assert players >= 2 and players <= 6, "Must be between 2 and 6 players."
        self.players = players
        
        # shuffle the initial tokens
        # independently shuffle the four classes of tokens
        level_1 = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        np.random.shuffle(level_1)
        level_2 = np.array([4, 4, 5, 5, 6, 6, 7, 7])
        np.random.shuffle(level_2)
        level_3 = np.array([8, 8, 9, 9, 10, 10, 11, 11])
        np.random.shuffle(level_3)
        level_4 = np.array([12, 12, 13, 13, 14, 14, 15, 15])
        np.random.shuffle(level_4)
        
        # stack groups into a single array
        self.tokens = np.concatenate((level_1, level_2, level_3, level_4), axis = 0)
        
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
    def drop_token(self, index_to_drop):       
        # place the token back in line
        self.tokens[self.positions[self.active_player][0]] = self.inventory[self.active_player][index_to_drop]
                                                                      
        # remove it from the player's inventory
        self.inventory[self.active_player] = np.delete(self.inventory[self.active_player], [index_to_drop])
        
    # bound function to start the next round
    def next_round(self):
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
                        selected_token = death_drop(categorize_tokens(self.inventory[j]))

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

## function take_turn
# takes a GameState plus Q tables and epsilon values and advances it by a turn
def take_turn(game, player_agent):
    # reporter variables
    turn_taken = False
    turn_around = np.nan
    pick_up = np.nan
    drop = np.nan
    pre_roll_gamestate = np.empty(0)
    post_roll_gamestate = np.empty(0)
    
    # check that the active player is not already back on the submarine
    if not (game.positions[game.active_player, 0] == -1 and game.positions[game.active_player, 1] == -1):
        # note that the turn was taken
        turn_taken = True
    
        # update the air supply
        game.update_air_supply()

        # determine whether the active player turns around
        if game.positions[game.active_player, 1] == 1 and game.positions[game.active_player, 0] != -1:
            if player_agent.turn_around_decision(game.pass_gamestate()):
                # turn the player back toward the sub if True
                game.positions[game.active_player, 1] = -1;
                turn_around = 1
            else:
                turn_around = 0
        
        # document the game state before rolling
        pre_roll_gamestate = game.pass_gamestate()
        
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
        
        # document the game state after rolling
        post_roll_gamestate = game.pass_gamestate()
        
        # determine whether the active player picks up or drops a token
        # check to see if a token is available
        if game.tokens[game.positions[game.active_player, 0]] > -1:
            if player_agent.pick_up_decision(game.pass_gamestate()):
                # pick up the token if True
                game.collect_token()
                pick_up = 1
            else:
                pick_up = 0
        elif game.tokens[game.positions[game.active_player, 0]] == -1:
            if len(game.inventory[game.active_player]) > 0:
                # select token to drop
                drop = player_agent.drop_decision(game.pass_gamestate())
                
                # drop a token if any selected
                if drop != 0:
                    # subtracting one to realign to zero-indexing
                    game.drop_token(drop - 1)
        
    # check to see if oxygen has run out
    if game.air_supply <= 0 or np.all(game.positions[:, 0] == -1):
        # start the next round
        game.next_round()
    
    # return a boolean indicating whether the game is over alongside descriptions of player's actions
    return(game.round <= 3, turn_taken, turn_around, pick_up, drop, pre_roll_gamestate, post_roll_gamestate)

## function simulate_game
# takes an integer number of players plus the player agent to use
# returns a dataframe of board states and a dataframe of player results
def simulate_game(players, player_agent):
    # initiate a new game state
    game = GameState(players)
    
    # declare arrays to document the gamestate
    state_turn_around = [np.empty(shape = (0, game.pass_gamestate().shape[0])) for i in range(players)]
    state_pick_up = [np.empty(shape = (0, game.pass_gamestate().shape[0])) for i in range(players)]
    state_drop_token = [np.empty(shape = (0, game.pass_gamestate().shape[0])) for i in range(players)]
    
    # declare arrays to document the player agent decisions
    action_turn_around = [np.empty(shape = (0, 1)) for i in range(players)] # happens prior to board update
    action_pick_up = [np.empty(shape = (0, 1)) for i in range(players)] # happens after board update
    action_drop_token = [np.empty(shape = (0, 1)) for i in range(players)] # happens after board update
    
    # declare a flag for game end
    continue_game = True
    
    # take turns until the game is over
    while continue_game:
        # take a turn, collect the player actions taken
        continue_game, turn_taken, turn_around, pick_up, drop, pre_roll_gamestate, post_roll_gamestate = take_turn(game, player_agent)
        
        # skip to next turn if active player is already back at the sub
        if turn_taken == False:
            # update the active player
            if game.active_player < game.players - 1:
                game.active_player += 1
            else:
                game.active_player = 0
            
            # next turn
            continue
        
        # document the game state
        state_turn_around[game.active_player] = np.vstack((state_turn_around[game.active_player], pre_roll_gamestate))
        state_pick_up[game.active_player] = np.vstack((state_pick_up[game.active_player], post_roll_gamestate))
        state_drop_token[game.active_player] = np.vstack((state_drop_token[game.active_player], post_roll_gamestate))
        
        # document the player decisions
        action_turn_around[game.active_player] = np.vstack((action_turn_around[game.active_player], turn_around))
        action_pick_up[game.active_player] = np.vstack((action_pick_up[game.active_player], pick_up))
        action_drop_token[game.active_player] = np.vstack((action_drop_token[game.active_player], drop))
        
        # update the active player
        if game.active_player < game.players - 1:
            game.active_player += 1
        else:
            game.active_player = 0
        
    # document the winner
    # ignoring tie breakers, all players lose when tied
    placement = np.zeros(shape = len(game.player_scores), dtype = int)
    if not np.all(game.player_scores == game.player_scores[0]):
        placement[np.argmax(game.player_scores)] = 1
    
    # return the arrays of gamestates and player actions 
    return(
        (state_turn_around, action_turn_around),
        (state_pick_up, action_pick_up),
        (state_drop_token, action_turn_around),
        placement
    )

## function compare_models
# takes two tuples of models and an integer number of games
# simulates games and returns the number of wins for each
def compare_models(player_agent_1, player_agent_2, games = 1000):
    # initiate counter to track win totals
    wins = np.zeros(2)
    
    for i in range(games):
        # initiate a new game state
        game = GameState(6)

        # start turn counter; declare a flag for game end
        turn = 0
        continue_game = True

        # randomly assign player_agent_1 and player_agent_2 to even/odd turns
        player_agent_1_turns = randint.rvs(0, 1)

        # take turns until the game is over
        while continue_game:
            # use player_agent_1 on half the turns
            if turn % 2 == player_agent_1_turns:
                # take a turn
                continue_game = take_turn(game, player_agent_1)[0]
                
            # use player_agent_2 on the other half
            else:
                # take a turn
                continue_game = take_turn(game, player_agent_2)[0]

            # increment turn counter
            turn += 1

        # document the winner    
        if np.argmax(game.player_scores) % 2 == player_agent_1_turns:
            wins[0] += 1
        else:
            wins[1] += 1
    
    # return the win totals
    return wins
