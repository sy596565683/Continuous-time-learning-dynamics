from wimblepong import Wimblepong
import random


class RandomAi(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm a very random AI. All I can play is Wimblepong.")
        self.env = env
        # Set the player id that determines on which side the ai is going to play
        self.player_id = player_id
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4
        self.name = "RandomAI"

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def get_action(self, ob=None):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """
        action = random.choice([self.env.MOVE_UP, self.env.MOVE_DOWN, self.env.STAY])
        return action

    def reset(self):
        # Nothing to done for now...
        return


