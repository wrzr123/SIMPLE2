from typing import List

from gym.spaces import Discrete
from numpy import ndarray

from wizard.envs.classes import Player, Trick
from wizard.envs.constants import DETERMINE_TRUMP, GUESS, PLAY, SUIT_ORDER, WIZARD, JESTER


class RuleBasedPlayer:
    def __init__(self,
                 players: List[Player],
                 current_player_num: int,
                 phase: str,
                 trick_guesses: List[int],
                 tricks: List[Trick],
                 current_trick: Trick,
                 action_space: Discrete,
                 legal_actions: ndarray):
        self.players = players
        self.current_player_num = current_player_num
        self.phase = phase
        self.trick_guesses = trick_guesses
        self.tricks = tricks
        self.current_trick = current_trick
        self.action_space = action_space
        self.legal_actions = legal_actions

    @property
    def player(self) -> Player:
        return self.players[self.current_player_num]

    def make_move(self) -> List[float]:
        if self.phase == DETERMINE_TRUMP:
            return self.determine_trump()
        elif self.phase == GUESS:
            return self.guess()
        elif self.phase == PLAY:
            return self.play()
        raise RuntimeError("Game is in unknown phase")

    def determine_trump(self) -> List[float]:
        suit_counts = [self.player.get_suit_count(suit) for suit in SUIT_ORDER]
        min_value = min(suit_counts)
        min_suit_index = suit_counts.index(min_value)
        return self.create_action_probs(min_suit_index)

    def guess(self) -> List[float]:
        wizard_count = sum(1 for card in self.player.cards if card.value == WIZARD)
        return self.create_action_probs(wizard_count )

    def play(self) -> List[float]:
        tricks_won = sum(1 for trick in self.tricks if trick.finished and trick.winner == self.player.id)
        guess = self.trick_guesses[self.player.id]
        first_to_act = self.current_trick.first_to_act
        # no further tricks to make
        if tricks_won >= guess:
            # not first to act
            if not first_to_act:
                card_index = self.get_highest_card_that_doesnt_win()
                if card_index > -1:
                    return self.create_action_probs(card_index)
                # no card was found that doesn't win
                # if not last to act, play the lowest possible card and hope somebody goes over
                if not self.current_trick.last_to_act:
                    return self.create_action_probs(self.get_lowest_card())
                # last to act, no hope anymore. Play the highest card possible to prevent further damage
                return self.create_action_probs(self.get_highest_card())
            # first to act
            else:
                return self.create_action_probs(self.get_lowest_card())
        # tricks to make
        else:
            # not first to act
            if not first_to_act:
                # still try to not win the trick, to wait until last round in best case
                card_index = self.get_highest_card_that_doesnt_win()
                if card_index > -1:
                    return self.create_action_probs(card_index)
                # if not possible to not win the trick, play the highest possible allowed card
                return self.create_action_probs(self.get_highest_card())
            # first to act
            else:
                return self.create_action_probs(self.get_highest_card())

    def create_action_probs(self, action) -> List[float]:
        phase_adjustment = 0
        if self.phase == GUESS: phase_adjustment = 4
        if self.phase == PLAY: phase_adjustment = 4 + 16
        action_probs = [0.01] * self.action_space.n
        action_probs[action + phase_adjustment] = 1 - 0.01 * (self.action_space.n - 1)
        return action_probs

    def get_highest_card_that_doesnt_win(self, except_wizard=False) -> int:
        legal_card_indices = self.get_legal_card_indices()
        cards_that_dont_win_indices: List[int] = []
        current_trick = self.current_trick
        for card_index in legal_card_indices:
            wins = current_trick.would_card_win_trick(self.player.cards[card_index])
            if not wins:
                cards_that_dont_win_indices.append(card_index)
        if len(cards_that_dont_win_indices) == 0: # means all cards win
            return -1
        if len(cards_that_dont_win_indices) == 1: # means there is only one card that doesn't win
            return cards_that_dont_win_indices[0]
        if not except_wizard: # maybe we don't want to throw away our wizard
            for card_index in cards_that_dont_win_indices:
                if self.player.cards[card_index].value == WIZARD:
                    return card_index # means we have a wizard which doesn't win and want to throw it away
        trump_card_indices: List[int] = []
        for card_index in cards_that_dont_win_indices:
            if self.player.cards[card_index].suit == current_trick.trump.suit and not self.player.cards[card_index].is_white_card:
                trump_card_indices.append(card_index)
        # in case we have trumps that don't win, throw away the highest trump
        if len(trump_card_indices) == 1:
            return trump_card_indices[0]
        if len(trump_card_indices) > 1:
            highest_trump_index = trump_card_indices[0]
            for i in range(1, len(trump_card_indices)):
                if self.player.cards[trump_card_indices[i]].value > self.player.cards[highest_trump_index].value:
                    highest_trump_index = trump_card_indices[i]
            return highest_trump_index
        # at this point there are only non-trump cards, jesters and maybe wizards left over
        # go ahead with all non-wizard and non-jester cards
        normal_cards_indices: List[int] = []
        for card_index in cards_that_dont_win_indices:
            if not self.player.cards[card_index].is_white_card:
                normal_cards_indices.append(card_index)
        if len(normal_cards_indices) == 1:
            return normal_cards_indices[0]
        if len(normal_cards_indices) > 1:
            # now check which is the card with the least remaining of its color. We want to play that to become color-free
            # if multiple colors have the same amount of cards left, play the highest rank.
            highest_card_index = normal_cards_indices[0]
            lowest_count = self.player.get_suit_count(self.player.cards[highest_card_index].suit)
            for i in range(1, len(normal_cards_indices)):
                count = self.player.get_suit_count(self.player.cards[normal_cards_indices[i]].suit)
                if count < lowest_count or (count == lowest_count and self.player.cards[normal_cards_indices[i]].value > self.player.cards[highest_card_index].value):
                    lowest_count = count
                    highest_card_index = normal_cards_indices[i]
            return highest_card_index
        # at this point there should only be jesters and wizards
        for i in cards_that_dont_win_indices:
            if self.player.cards[i].value == JESTER:
                return i
        return cards_that_dont_win_indices[0]

    def get_lowest_card(self) -> int:
        legal_card_indices = self.get_legal_card_indices()
        for card_index in legal_card_indices:
            if self.player.cards[card_index].value == JESTER:
                return card_index
        normal_card_indices: List[int] = []
        current_trick = self.current_trick
        for card_index in legal_card_indices:
            if self.player.cards[card_index].suit != current_trick.trump.suit and self.player.cards[card_index].value != WIZARD:
                normal_card_indices.append(card_index)
        if len(normal_card_indices) == 1:
            return normal_card_indices[0]
        if len(normal_card_indices) > 1:
            lowest_value_index = normal_card_indices[0]
            for i in range(1, len(normal_card_indices)):
                if self.player.cards[normal_card_indices[i]].value < self.player.cards[lowest_value_index].value:
                    lowest_value_index = normal_card_indices[i]
            # Could be improved: In case there are multiple cards with the same value, choose the one with fewer of their suit left
            return lowest_value_index
        # only trumps and wizards left
        trump_card_indices: List[int] = []
        for card_index in legal_card_indices:
            if not self.player.cards[card_index].is_white_card:
                trump_card_indices.append(card_index)
        if len(trump_card_indices) == 1:
            return trump_card_indices[0]
        if len(trump_card_indices) > 1:
            lowest_card_index = trump_card_indices[0]
            for i in range(1, len(trump_card_indices)):
                if self.player.cards[trump_card_indices[i]].value < self.player.cards[lowest_card_index].value:
                    lowest_card_index = trump_card_indices[i]
            return lowest_card_index
        # only wizards left
        return legal_card_indices[0]

    def get_highest_card(self) -> int:
        legal_card_indices = self.get_legal_card_indices()
        # search for wizards
        for card_index in legal_card_indices:
            if self.player.cards[card_index].value == WIZARD:
                return card_index
        # search for trumps
        trump_card_indices: List[int] = []
        current_trick = self.current_trick
        for card_index in legal_card_indices:
            if self.player.cards[card_index].suit == current_trick.trump.suit and not self.player.cards[card_index].is_white_card:
                trump_card_indices.append(card_index)
        if len(trump_card_indices) == 1:
            return trump_card_indices[0]
        if len(trump_card_indices) > 1:
            highest_card_index = trump_card_indices[0]
            for i in range(1, len(trump_card_indices)):
                if self.player.cards[trump_card_indices[i]].value > self.player.cards[highest_card_index].value:
                    highest_card_index = trump_card_indices[i]
            return highest_card_index
        # search for normal cards (all cards except jesters)
        normal_card_indices: List[int] = []
        for card_index in legal_card_indices:
            if self.player.cards[card_index].value != JESTER:
                normal_card_indices.append(card_index)
        if len(normal_card_indices) == 1:
            return normal_card_indices[0]
        if len(normal_card_indices) > 1:
            highest_value_index = normal_card_indices[0]
            for i in range(1, len(normal_card_indices)):
                if self.player.cards[normal_card_indices[i]].value > self.player.cards[highest_value_index].value:
                    highest_value_index = normal_card_indices[i]
            # Could be improved: In case there are multiple cards with the same value, choose the one with fewer of their suit left
            return highest_value_index
        # only jesters left
        return legal_card_indices[0]

    def get_legal_card_indices(self) -> List[int]:
        ret = []
        legal_actions = self.legal_actions
        for i in range(len(self.player.cards)):
            if legal_actions[i + 4 + 16] == 1:
                ret.append(i)
        return ret