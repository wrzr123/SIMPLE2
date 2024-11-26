import math

import gym
import numpy as np
from typing import List, Tuple
import random

from numpy import ndarray
from stable_baselines import logger

DIFFERENT_CARD_VALUES = 8
RED = '🔴'
BLUE = '🔵'
GREEN = '🟢'
YELLOW = '🟡'
SUIT_ORDER = [RED, BLUE, GREEN, YELLOW]

class Card:
    def __init__(self, value: int, suit: int):
        self.value = value
        self.suit = suit
        self.id = value + suit * DIFFERENT_CARD_VALUES
        self.suit_str = SUIT_ORDER[suit]

    def __repr__(self):
        return f"{self.suit_str}{self.value + 1}"
        

DECK: List[Card] = [
    Card(0, 0),
    Card(1, 0),
    Card(2, 0),
    Card(3, 0),
    Card(4, 0),
    Card(5, 0),
    Card(6, 0),
    Card(7, 0),
    Card(0, 1),
    Card(1, 1),
    Card(2, 1),
    Card(3, 1),
    Card(4, 1),
    Card(5, 1),
    Card(6, 1),
    Card(7, 1),
    Card(0, 2),
    Card(1, 2),
    Card(2, 2),
    Card(3, 2),
    Card(4, 2),
    Card(5, 2),
    Card(6, 2),
    Card(7, 2),
    Card(0, 3),
    Card(1, 3),
    Card(2, 3),
    Card(3, 3),
    Card(4, 3),
    Card(5, 3),
    Card(6, 3),
    Card(7, 3)
]


class Player:
    def __init__(self, index: int):
        self.index = index
        self.cards: List[Card] = []

    def has_suit(self, suit: int) -> bool:
        for card in self.cards:
            if card.suit == suit:
                return True
        return False


class Play:
    def __init__(self, player_id: int, card: Card):
        self.player_id: int = player_id
        self.card: Card = card


class Trick:
    def __init__(self, n_players: int, trump_suit: int):
        self.plays: List[Play] = []
        self.n_players = n_players
        self.trump_suit = trump_suit

    def add_play(self, play: Play):
        self.plays.append(play)

    @property
    def finished(self) -> bool:
        return len(self.plays) == self.n_players

    @property
    def winning_play(self) -> Play:
        highest_card = None
        winning_play = None
        for play in self.plays:
            if highest_card is None:
                highest_card = play.card
                winning_play = play
                continue
            if self.check_if_card_is_higher(highest_card, play.card):
                highest_card = play.card
                winning_play = play
        return winning_play

    @property
    def winner(self) -> int:
        return self.winning_play.player_id

    @property
    def first_to_act(self) -> bool:
        return len(self.plays) == 0

    @property
    def last_to_act(self) -> bool:
        return len(self.plays) == self.n_players - 1

    @property
    def first_card(self) -> Card:
        return self.plays[0].card

    @property
    def free_to_play(self) -> bool:  # returns true if all cards are allowed to be played
        return self.first_to_act

    def is_card_allowed_to_be_played(self, card):
        return self.free_to_play or card.suit == self.first_card.suit

    def check_if_card_is_higher(self, lower: Card, higher: Card) -> bool:
        if higher.suit != lower.suit:
            return higher.suit == self.trump_suit
        return higher.value > lower.value

    def __repr__(self):
        return ', '.join(map(str, (play.card for play in self.plays)))


class BasicTrickGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False, n_players = 4):
        super(BasicTrickGameEnv, self).__init__()
        self.name = 'basictrickgame'
        self.n_players = n_players
        self.verbose = verbose
        self.manual = manual

        self.deck: List[Card] = []
        self.players: List[Player] = []
        self.current_player_num = 0
        self.tricks: List[Trick] = []
        self.tricks_won: List[int] = []
        self.max_tricks = math.floor(len(DECK) / n_players)
        self.current_trick_num = 0
        self.current_starting_player_num = 0
        self.trump_suit = 3
        self.done = False

        self.action_space = gym.spaces.Discrete(len(DECK))
        self.hand_size = len(DECK)
        self.trick_size = len(DECK) * n_players
        self.played_cards_size = len(DECK)
        self.trump_suit_size = 4
        self.trump_enabled_size = 1
        self.maximize_tricks_size = 1
        legal_actions_size = self.action_space.n
        total_observation_size = (self.hand_size + self.trick_size + self.played_cards_size + self.trump_suit_size
                                  + self.maximize_tricks_size + self.trump_enabled_size + legal_actions_size)
        self.observation_space = gym.spaces.Box(-1, 1, (total_observation_size,), dtype=np.int8)

    @property
    def observation(self) -> ndarray:

        # Represent the player's hand as a one-hot vector per card
        hand_obs = np.full(self.hand_size, -1, dtype=np.int8)
        for card in self.current_player.cards:
            hand_obs[card.id] = 1

        # Represent the cards already played in the trick as a one-hot vector per card per trick
        trick_obs = np.zeros(self.trick_size, dtype=np.int8)
        amount_of_cards = len(DECK)
        for i, play in enumerate(self.current_trick.plays):
            # Set all bits in the 60-bit block to -1 for the current card
            trick_obs[i * amount_of_cards: (i + 1) * amount_of_cards] = -1
            # Set the specific bit for the played card to 1
            trick_obs[i * amount_of_cards + play.card.id] = 1

        # Represent the cards already played in previous trick as a one-hot vector per card
        played_cards_obs = np.full(self.played_cards_size, -1, dtype=np.int8)
        for trick in self.tricks:
            for play in trick.plays:
                played_cards_obs[play.card.id] = 1

        # Represent the trump-suit as one-hot vector and trump enabled flag
        if self.trump_suit > -1:
            trump_suit_obs = np.full(self.trump_suit_size, -1, dtype=np.int8)
            trump_suit_obs[self.trump_suit] = 1
            trump_enabled_obs = np.full(self.trump_enabled_size, 1, dtype=np.int8)
        else:
            trump_suit_obs = np.full(self.trump_suit_size, 0, dtype=np.int8)
            trump_enabled_obs = np.full(self.trump_enabled_size, -1, dtype=np.int8)

        # Minimize or maximize tricks
        maximize_tricks_obs = np.full(self.maximize_tricks_size, 1, dtype=np.int8)

        # Add legal actions
        legal_actions = self.legal_actions

        # Combine all observations into a single vector
        observation = np.concatenate([hand_obs, trick_obs, played_cards_obs, trump_suit_obs, trump_enabled_obs, maximize_tricks_obs, legal_actions])

        return observation

    @property
    def legal_actions(self) -> ndarray:
        actions = np.zeros(self.action_space.n, dtype=np.int8)
        available_cards = self.current_player.cards
        if self.current_trick.first_to_act or not self.current_player.has_suit(self.current_trick.first_card.suit):
            for card in available_cards:
                actions[card.id] = 1
        else:
            for card in available_cards:
                if self.current_trick.is_card_allowed_to_be_played(card):
                    actions[card.id] = 1
        return actions

    def step(self, action: int) -> Tuple[ndarray, List[int], bool, dict]:
        reward = [0] * self.n_players

        # check move legality
        if self.legal_actions[action] == 0:
            reward[self.current_player_num] = -100
            return self.observation, reward, True, {}

        if self.done:
            raise ValueError("Game is already finished.")

        card_index = next(i for i, card in enumerate(self.current_player.cards) if card.id == action)
        play = Play(self.current_player.index, self.current_player.cards.pop(card_index))
        self.current_trick.add_play(play)
        logger.debug(f"Player {self.current_player_num + 1} plays {play.card} (Trick: {self.current_trick})")
        self.current_player_num = (self.current_player_num + 1) % self.n_players

        if self.current_player_num == self.current_starting_player_num:
            logger.debug(f"Trick {self.current_trick_num + 1} finished")
            logger.debug(f"Player {self.current_trick.winner + 1} won the trick")
            self.tricks_won[self.current_trick.winner] += 1
            if self.current_trick_num == self.max_tricks - 1:
                self.done = True
                return self.observation, self.score_game(), self.done, {}
            else:
                self.current_starting_player_num = self.current_trick.winner
                self.current_player_num = self.current_starting_player_num
                self.current_trick_num += 1
                self.tricks.append(Trick(self.n_players, self.trump_suit))

        return self.observation, [0] * self.n_players, self.done, {}

    def score_game(self) -> List[int]:
        rewards = [0] * self.n_players
        for trick in self.tricks:
            rewards[trick.winner] += 1
        return rewards

    def reset(self):
        self.deck = self.create_deck()
        self.players = [Player(index=i) for i in range(self.n_players)]
        self.current_player_num = 0
        self.shuffle_and_deal()
        random_number = random.uniform(0, 1)
        if random_number > 0.3:
            self.trump_suit = 3
        else:
            self.trump_suit = -1
        self.tricks = [Trick(self.n_players, self.trump_suit)]
        self.tricks_won = [0] * self.n_players
        self.current_trick_num = 0
        self.current_starting_player_num = 0
        self.done = False

    def render(self, mode='human', close=False):
        if close:
            return
        for player in self.players:
            logger.debug(f"Player {player.index + 1} holds {', '.join(map(str, player.cards))}")
        if self.trump_suit == -1:
            logger.debug(f"Current trump: no trump")
        else:
            logger.debug(f"Current trump: {SUIT_ORDER[self.trump_suit]}")
        if self.done:
            logger.debug("Game Over")
            logger.debug(f"Final rewards: {self.score_game()}")
        else:
            logger.debug(f"Player {self.current_player_num + 1}'s turn")

    def rules_move(self):
        return NotImplementedError

    @staticmethod
    def create_deck() -> List[Card]:
        return [Card(card.value, card.suit) for card in DECK]

    def shuffle_and_deal(self):
        random.shuffle(self.deck)
        card_count = len(self.deck)
        player_num = self.current_player_num
        for _ in range(card_count):
            self.players[player_num].cards.append(self.deck.pop())
            player_num = (player_num + 1) % self.n_players

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]

    @property
    def current_trick(self) -> Trick:
        return self.tricks[self.current_trick_num]

    def parse_human_action(self, action: str) -> int:
        try:
            card_index = int(action) - 1
            return self.players[self.current_player_num].cards[card_index].id
        except:
            logger.debug(f"Invalid card '{action}'")

        return -1

    def to_human_action(self, action: int) -> str:
        if action > len(DECK) - 1 or action < 0: return f"Invalid move '{action}'"
        return str(DECK[action])