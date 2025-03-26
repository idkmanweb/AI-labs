import random
from exceptions import AgentException
import copy


class MinMaxAgent:
    def __init__(self, my_token='x'):
        self.my_token = my_token

    def m(self, connect4, x, depth):
        if depth == 0:
            res = 0
            return res, random.choice(connect4.possible_drops())

        if connect4.game_over:
            if connect4.wins == self.my_token:
                res = 1
            elif connect4.wins is None:
                res = 0
            else:
                res = -1

            return res, random.choice(connect4.possible_drops())

        else:
            all_m = []
            if x == 1:
                best_m = -1
                for drop in connect4.possible_drops():
                    connect4copy = copy.deepcopy(connect4)
                    connect4copy.drop_token(drop)
                    s_res, _ = self.m(connect4copy, 0, depth-1)
                    if s_res > best_m:
                        best_m = s_res
                        all_m = [drop]
                    elif s_res == best_m:
                        all_m.append(drop)
            else:
                best_m = 1
                for drop in connect4.possible_drops():
                    connect4copy = copy.deepcopy(connect4)
                    connect4copy.drop_token(drop)
                    s_res, _ = self.m(connect4copy, 1, depth-1)
                    if s_res < best_m:
                        best_m = s_res
                        all_m = [drop]
                    elif s_res == best_m:
                        all_m.append(drop)

            res = best_m
            move = random.choice(all_m)

        return res, move

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        _, move = self.m(connect4, 1, 5)

        return move
