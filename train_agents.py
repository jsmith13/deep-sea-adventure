### This script trains the player agent.

# training parameters
players = 6
games_per_cycle = 50
cycles_per_target_update = 10

# import required libraries and scripts
import agents
import game_functions

# initiate a player agent to train
training_agent = agents.PlayerAgent(players)

# declare a counter for number of complete update cycles
update = 1

# train the agent until stopped
while True:
    # count the number of training cycles
    for i in range(cycles_per_target_update):
        # simulate games and store them in memory    
        for j in range(games_per_cycle):
            game_log = game_functions.simulate_game(players, training_agent)
            training_agent.store_game(game_log)
        
            # train the agent using the simulated games
            training_agent.train_model()
            
        # update console
        print("Training cycle " + str(i + 1) + " of " + str(cycles_per_target_update))
        
    # update the target weights
    training_agent.update_target_weights()
    
    # save the current model
    training_agent.save_models(identifier = str(update))
    
    # update the exploration/exploitation ratio
    training_agent.epsilon *= training_agent.epsilon_decay
    if training_agent.epsilon < training_agent.epsilon_min:
        training_agent.epsilong = training_agent.epsilon_min
    
    # update console and counter
    print("Updated target weights. Update #" + str(update))
    print("Updated epsilon. New value " + str(training_agent.epsilon))
    update += 1
    