import ludopy
from matplotlib.pyplot import pie
import numpy as np

from GA_ai import Network, Population
from state_space_rep import StateSpace
from logger import Logger


def run_game(network):
    # Game setup
    g = ludopy.Game()
    there_is_a_winner = False
    stateSpace = StateSpace()
    
    if_ai_won = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            
            if (player_i == 0):
                scores = []
            
                # Compute network result for each movable piece
                # for movable_piece in move_pieces:
                    # piece = np.array(player_pieces[movable_piece])
                    
                    # enemies = np.array(enemy_pieces)
                    # enemies_sorted = []
                    # for enemy in enemies:
                    #     enemies_sorted.append(np.sort(enemy))
                    # enemies = np.array(enemies_sorted)
                    
                    # input = np.resize(np.concatenate((dice, piece, enemies), axis=None), (14, 1))
                    # scores.append(network.compute_result(input))
                    
                    
                input = stateSpace.get_state(dice, player_pieces, enemy_pieces)
                scores = network.compute_result(input)
                
                highscore = 0
                move_piece = move_pieces[0]
                
                for piece in move_pieces:
                    if (scores[piece] > highscore):
                        highscore = scores[piece]
                        move_piece = piece
                
                piece_to_move = move_piece
                # piece_to_move = move_pieces[np.argmax(scores)]
                # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            
            else:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        
        if (there_is_a_winner and player_i == 0):
            if_ai_won = True
        
    # print("Saving history to numpy file")
    # g.save_hist(f"game_history.npy")
    # print("Saving game video")
    # g.save_hist_video(f"game_video.mp4")
    
    return if_ai_won


if __name__ == '__main__':
    network = Network(44, 4, 2, 8)
    
    run_game(network)