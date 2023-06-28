import re
import numpy as np

from ludopy import player


class StateSpace:
    def __init__(self):
        self.enemy_pieces = []
        self.state = []

    def get_state(self, die, player_pieces, enemy_pieces):
        self.enemy_pieces = np.array(enemy_pieces)
        player_pieces = np.array(player_pieces)

        for piece in player_pieces:
            self.state.append(self.canActivate(die, piece))
            self.state.append(self.canGoal(die, piece))
            self.state.append(self.canKill(die, piece))
            self.state.append(self.canStar(die, piece))
            self.state.append(self.willbeSection1(die, piece))
            self.state.append(self.willbeSection2(die, piece))
            self.state.append(self.willbeSection3(die, piece))
            self.state.append(self.willbeSection4(die, piece))
            self.state.append(self.willbeDanger(die, piece))
            self.state.append(self.willbeKilled(die, piece))
            self.state.append(self.willbeHomeStretch(die, piece))
            self.state.append(self.willbeSafe(die, piece))

        # return np.array(self.state)   # Right formatting for ANN input
        return np.resize(
            np.array(self.state), (48, 1)
        )  # Right formatting for ANN input

    ### Oppotunities ###

    def canActivate(self, die, piece_position):
        if piece_position != 0:
            return False
        if die == 6:
            return True
        else:
            return False

    def canGoal(self, die, piece_position):
        position = piece_position + die
        if position == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        if position == player.GOAL_INDEX:
            return True
        else:
            return False

    def canKill(self, die, piece_position):
        if self.willbeKilled(die, piece_position):
            return False
        differences = self.enemy_pieces - piece_position
        kill = np.any(differences == die)
        return kill

    def canStar(self, die, piece_position):
        position = piece_position + die
        if position == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        star = np.any(np.array(player.STAR_INDEXS) == position)
        return star

    ### Consequences ###

    def willbeSection1(self, die, piece_position):
        section = range(1, 14)
        return (piece_position + die) in section

    def willbeSection2(self, die, piece_position):
        section = range(14, 27)
        return (piece_position + die) in section

    def willbeSection3(self, die, piece_position):
        section = range(27, 40)
        return (piece_position + die) in section

    def willbeSection4(self, die, piece_position):
        section = range(40, 53)
        return (piece_position + die) in section

    def willbeDanger(self, die, piece_position):
        if self.willbeSafe(die, piece_position) == True:
            return False
        if self.willbeKilled(die, piece_position) == True:
            return True

        position = piece_position + die

        enemy_globes = [
            player.ENEMY_1_GLOB_INDX,
            player.ENEMY_2_GLOB_INDX,
            player.ENEMY_3_GLOB_INDX,
        ]
        if np.any(np.array(enemy_globes) == position):
            return True

        if position > 6:
            differences = (position) - self.enemy_pieces
            killzone = ((0 < differences) & (differences <= 6)).any()
        else:  # TODO: Fix! (wrap around - ex. 53 is a danger to 1-6)
            differences = (position) - self.enemy_pieces
            killzone = ((0 < differences) & (differences <= 6)).any()

        return killzone

    def willbeKilled(self, die, piece_position):
        # TODO: land on enemy home globe with correct enemy

        # Check for duplicates on one field
        unique, counts = np.unique(self.enemy_pieces, return_counts=True)
        lookup = dict(zip(unique, counts))
        try:
            return lookup[piece_position + die] >= 2
        except:
            return False

    def willbeHomeStretch(self, die, piece_position):
        if piece_position + die >= 53:
            return True
        if piece_position + die == player.STAR_AT_GOAL_AREAL_INDX:
            return True
        else:
            return False

    def willbeSafe(self, die, piece_position):
        if self.willbeHomeStretch(die, piece_position) == True:
            return True
        position = piece_position + die
        safe = np.any(np.array(player.GLOB_INDEXS) == position)
        safe |= player.START_INDEX
        return safe


if __name__ == "__main__":
    state = StateSpace()

    print(
        state.get_state(
            2,
            [0, 0, 7, 30],
            np.array([[52, 3, 7, 35], [43, 3, 4, 36], [6, 8, 2, 35], [52, 3, 7, 35]]),
        )
    )
    # print(state.canStar(2, 36))
