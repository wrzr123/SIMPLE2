from typing import List

import gym
import numpy as np
import random
from stable_baselines import logger

from wizard.envs.classes import Player, Card, Trump, Trick, Play
from wizard.envs.constants import JESTER, WIZARD, GUESS, SUIT_ORDER, DETERMINE_TRUMP, GAME_PHASES, PLAY
from wizard.envs.rule_based_player import RuleBasedPlayer

class WizardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False, n_players = 4):
        super(WizardEnv, self).__init__()
        self.name = 'wizard'
        self.n_players = n_players
        self.verbose = verbose
        self.manual = manual

        self.current_round = 4 # is used with zero-based index
        self.deck = self.create_deck()
        self.players = [Player(id=i) for i in range(n_players)]
        self.trick_guesses = [-1] * n_players  # Initialize guesses to an invalid state
        self.tricks_won = [0] * n_players
        self.trump = None
        self.tricks = []
        self.current_trick_num = 0
        self.current_starting_player_num = 0

        # Define Gym action space
        self.action_space = gym.spaces.Discrete(4 + 16 + 15)  # Player can guess 0-15 tricks; player can play card 1-15 on his hand
        # Define Gym observation space
        phase_size = 3 # 3 phases determine trump, guess, play
        hand_size = (self.current_round + 1) * (15 + 4) # (all possible cards)
        trick_size = self.n_players * (15 + 4) # (all possible cards)
        trump_size = 4 # (4 suits)
        position_size = self.n_players
        guess_size = self.n_players * (self.current_round + 2) # (guesses for current round)
        tricks_won_size = self.current_round + 2
        legal_actions_size = self.action_space.n
        total_observation_size = phase_size + hand_size + trick_size + trump_size + position_size + guess_size + tricks_won_size + legal_actions_size
        self.observation_space = gym.spaces.Box(0, 1, (total_observation_size,), dtype=np.int8)

        self.current_player_num = 0
        self.dealer_num = n_players - 1
        self.phase = GUESS  # Phases: 'determine_trump', 'guess' or 'play'
        self.done = False

    def create_deck(self):
        suits = SUIT_ORDER
        return [Card(value, suit) for value in range(0, 15) for suit in suits]

    def shuffle_and_deal(self):
        random.shuffle(self.deck)
        for _ in range(self.current_round + 1):
            for player in self.players:
                player.cards.append(self.deck.pop())
                #logger.debug(f"Player {player.id} gets dealt {player.cards[len(player.cards) - 1]}")
        if len(self.deck) > 0:
            trump_card = self.deck.pop()
            self.trump = Trump(trump_card)
            if self.trump.card.value == WIZARD:
                self.phase = DETERMINE_TRUMP
                self.current_player_num = self.dealer_num
        self.tricks.append(Trick(self.trump, self.n_players))

    # Create an observation for the current state of the game
    @property
    def observation(self):
        player = self.players[self.current_player_num]

        # # Represent the current round phase as scalar
        # phase_obs = np.ndarray(1, dtype=np.float32)
        # phase_obs[0] = self.value_to_scalar_base_array(GAME_PHASES.index(self.phase), GAME_PHASES)

        # Represent the current round phase as one-hot encoded vector
        phase_obs = np.zeros(3, dtype=np.int8)
        phase_obs[GAME_PHASES.index(self.phase)] = 1

        # # Represent the player's hand as a 30-element scalar array
        # # 2 elements per card, one for the value, one for the suit
        # hand_obs = np.full(30, -1, dtype=np.float32)
        # for i, card in enumerate(player.cards):
        #     hand_obs[2 * i] = self.card_value_to_scalar(card)
        #     hand_obs[2 * i + 1] = self.card_suit_to_scalar(card)

        # Represent the player's hand as a one-hot vector per card and suit for each card
        hand_obs = np.zeros((self.current_round + 1) * (15 + 4), dtype=np.int8)
        for i, card in enumerate(player.cards):
            hand_obs[i * 19 + card.value] = 1
            hand_obs[i * 19 + 15 + SUIT_ORDER.index(card.suit)] = 1

        # # Represent the current trick as a scalar array
        # trick_obs = np.full(self.n_players * 2, -1, dtype=np.float32)
        # for i, play in enumerate(self.current_trick.plays):
        #     trick_obs[2 * i] = self.card_value_to_scalar(play.card)
        #     trick_obs[2 * i + 1] = self.card_suit_to_scalar(play.card)

        # Represent the current trick as a one-hot vector per card and suit for each card
        trick_obs = np.zeros(self.n_players * (15 + 4), dtype=np.int8)
        for i, play in enumerate(self.current_trick.plays):
            trick_obs[i * 19 + play.card.value] = 1
            trick_obs[i * 19 + 15 + SUIT_ORDER.index(play.card.suit)] = 1

        # # Represent the player's position as scalar
        # position_obs = np.ndarray(1, dtype=np.float32)
        # position_obs[0] = self.value_to_scalar_base_value(self.current_player_position, self.n_players)

        # Represent the player's position as a one-hot vector
        position_obs = np.zeros(self.n_players, dtype=np.int8)
        position_obs[self.current_player_position] = 1

        # # Represent the trump as scalar
        # trump_obs = np.full(1, -1, dtype=np.float32)
        # if self.trump and self.trump.suit:
        #     trump_obs[0] = self.value_to_scalar_base_array(SUIT_ORDER.index(self.trump.suit), SUIT_ORDER)

        # One-hot encode the trump suit (4 elements)
        trump_obs = np.zeros(4, dtype=np.int8)
        if self.trump and self.trump.suit:
            trump_obs[SUIT_ORDER.index(self.trump.suit)] = 1

        # # Add players' guesses as a scalar array (n_players elements), using relative positions
        # guesses_obs = np.full(self.n_players, -1, dtype=np.float32)
        # player_num = self.current_starting_player_num
        # for i in range(self.n_players):
        #     if self.trick_guesses[player_num] > -1:
        #         guesses_obs[player_num] = self.value_to_scalar_base_value(self.trick_guesses[player_num], 15, True)
        #     player_num = (player_num + 1) % self.n_players

        # Add players' guesses as one-hot vectors (n_players elements), using relative positions
        guesses_obs = np.zeros(self.n_players * (self.current_round + 2), dtype=np.int8)
        player_num = self.current_starting_player_num
        for i in range(self.n_players):
            if self.trick_guesses[player_num] > -1:
                guesses_obs[(self.current_round + 2) * i + self.trick_guesses[player_num]] = 1
            player_num = (player_num + 1) % self.n_players

        # # Represent the tricks made so far by player
        # tricks_won_obs = np.ndarray(1, dtype=np.float32)
        # tricks_won_obs[0] = self.value_to_scalar_base_value(self.tricks_won[self.current_player_num], 15, True)

        # Represent the tricks made so far by player
        tricks_won_obs = np.zeros(self.current_round + 2, dtype=np.int8)
        tricks_won_obs[self.tricks_won[self.current_player_num]] = 1

        # Add legal actions
        legal_actions = self.legal_actions

        # Combine all observations into a single vector
        observation = np.concatenate([phase_obs, hand_obs, trick_obs, position_obs, trump_obs, guesses_obs, tricks_won_obs, legal_actions])

        return observation

    @property
    def current_trick(self) -> Trick:
        return self.tricks[self.current_trick_num]

    @property
    def current_player_position(self) -> int:
        return (self.current_player_num  - self.current_starting_player_num) % self.n_players

    @property
    def legal_actions(self):
        # Determine legal actions based on the current phase
        actions = np.zeros(self.action_space.n, dtype=np.int8)
        if self.phase == DETERMINE_TRUMP:
            actions[0] = 1 # Player is allowed to say something between 0 and 3, representing the 4 colours
            actions[1] = 1
            actions[2] = 1
            actions[3] = 1
            return actions
        elif self.phase == GUESS:
            for i in range(self.current_round + 2): # i+2 because current_round is zero-based and 0 is always a valid guess
                actions[i + 4] = 1
            return actions
        else:
            if not self.done:
                player = self.players[self.current_player_num]
                for i, card in enumerate(player.cards):
                    # logger.debug(f"Checking for player {self.current_player_num} if card {card} is allowed to be played...")
                    if self.current_trick.is_card_allowed_to_be_played(card):
                        actions[i + 4 + 16] = 1
                    elif not player.has_suit(self.current_trick.first_card.suit):
                        actions[i + 4 + 16] = 1
                    # logger.debug(f"It's {'not ' if actions[i] == 0 else ''}allowed")
            return actions

    def card_to_scalar(self, card):
        # Convert a card to a unique index in the observation array
        suit_index = SUIT_ORDER.index(card.suit)
        return (suit_index * 15 + card.value + 1) / 60  # +1 adjusts card values from 1 to 15, so that 0 can mean no card

    def card_value_to_scalar(self, card):
        # Convert a card value to a unique index in the observation array
        return (card.value + 1) / 60  # +1 adjusts card values from 1 to 15, so that 0 can mean no card

    def card_suit_to_scalar(self, card):
        # Convert a card suit to a unique index in the observation array
        suit_index = SUIT_ORDER.index(card.suit)
        return self.value_to_scalar_base_array(suit_index, SUIT_ORDER)

    def card_to_index(self, card):
        # Convert a card to a unique index in the observation array
        suit_index = SUIT_ORDER.index(card.suit)
        return suit_index * 15 + card.value  # Value is already in the range 0-14

    def value_to_scalar_base_array(self, value: int, base: List):
        return self.value_to_scalar_base_value(value, len(base))

    def value_to_scalar_base_value(self, value: int, base: int, zero_based = False):
        if zero_based: return value / base
        return (value + 1) / base

    def step(self, action):
        # check move legality
        if self.legal_actions[action] == 0:
            reward = [0] * self.n_players
            reward[self.current_player_num] = -100
            return self.observation, reward, True, {}

        # perform action
        if self.phase == DETERMINE_TRUMP:
            self.trump.determined_suit = SUIT_ORDER[action]
            logger.debug(f"Player {self.current_player_num} determines {self.trump.suit} as trump suit")
            self.phase = GUESS
            self.current_player_num = (self.current_player_num + 1) % self.n_players

        elif self.phase == GUESS:
            self.trick_guesses[self.current_player_num] = (action - 4)  # Player makes a guess
            logger.debug(f"Player {self.current_player_num} guesses {(action - 4) } tricks.")
            self.current_player_num = (self.current_player_num + 1) % self.n_players

            # If all players have made their guesses, transition to the play phase
            if self.current_player_num == self.current_starting_player_num:
                self.phase = PLAY

        elif self.phase == PLAY:
            if self.done:
                raise ValueError("Game is already finished.")

            player = self.players[self.current_player_num]
            play = Play(player.id, player.cards.pop(action - 4 - 16))
            self.current_trick.add_play(play)
            logger.debug(f"Player {self.current_player_num} plays {play.card} (Trick: {self.current_trick})")
            self.current_player_num = (self.current_player_num + 1) % self.n_players

            if self.current_player_num == self.current_starting_player_num:
                logger.debug(f"Trick {self.current_trick_num + 1} finished")
                logger.debug(f"Player {self.current_trick.winner} won the trick")
                self.tricks_won[self.current_trick.winner] += 1
                if self.current_trick_num == self.current_round:
                    self.done = True
                    return self.observation, self.score_game(), self.done, {}
                else:
                    self.current_starting_player_num = self.current_trick.winner
                    self.current_player_num = self.current_starting_player_num
                    self.current_trick_num += 1
                    self.tricks.append(Trick(self.trump, self.n_players))

        return self.observation, [0] * self.n_players, self.done, {}

    def score_game(self):
        # Calculate normal wizard points
        points = [0] * self.n_players
        max_points = -999
        for i, guess in enumerate(self.trick_guesses):
            if guess == self.tricks_won[i]:
                points[i] = 20 + 10 * guess
            else:
                points[i] = -10 * abs(guess - self.tricks_won[i])
            if points[i] > max_points:
                max_points = points[i]

        return points

        # #Calculate reward
        # reward = [0] * self.n_players
        # for i, points in enumerate(points):
        #     if points > 0 and points == max_points:
        #         reward[i] = 1
        #     elif points > 0:
        #         reward[i] = 0
        #     else:
        #         reward[i] = -1
        #
        # return reward

    def reset(self):
        self.players = [Player(id=i) for i in range(self.n_players)] # needs to be reset in case game was cancelled due to invalid move
        self.current_player_num = 0
        self.dealer_num = self.n_players - 1
        self.trump = None
        self.phase = GUESS
        self.tricks = []
        self.current_trick_num = 0
        self.current_starting_player_num = 0
        self.deck = self.create_deck()
        self.shuffle_and_deal()
        self.trick_guesses = [-1] * self.n_players
        self.tricks_won = [0] * self.n_players
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        if close:
            return
        for player in self.players:
            logger.debug(f"Player {player.id} holds {', '.join(map(str, player.cards))}, guessed: {self.trick_guesses[player.id]}")
        logger.debug(f"Trump card: {self.trump}")
        if self.done:
            logger.debug("Game Over")
            logger.debug(f"Final rewards: {self.score_game()}")
        else:
            logger.debug(f"Phase: {self.phase}. Player {self.current_player_num}'s turn")

    def rules_move(self) -> List[float]:
        rule_player = RuleBasedPlayer(self.players, self.current_player_num, self.phase, self.trick_guesses, self.tricks, self.current_trick, self.action_space, self.legal_actions)
        return rule_player.make_move()