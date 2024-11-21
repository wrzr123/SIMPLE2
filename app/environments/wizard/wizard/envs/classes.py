from typing import List

from wizard.envs.constants import JESTER, WIZARD


class Card:
    def __init__(self, value, suit):
        self.value = value  # Numerical value of the card (1-13), 0 is jester, 14 is wizard
        self.suit = suit  # Suit of the card

    @property
    def is_white_card(self):
        return self.value == JESTER or self.value == WIZARD

    def __repr__(self):
        if self.value == WIZARD:
            return 'Wizard'
        if self.value == JESTER:
            return 'Jester'
        return f"{self.suit} {self.value}"

class Player:
    def __init__(self, id):
        self.id = id
        self.cards: List[Card] = []

    def has_suit(self, suit):
        for card in self.cards:
            if card.suit == suit and not card.is_white_card:
                return True
        return False

    def get_suit_count(self, suit) -> int:
        count = 0
        for card in self.cards:
           if card.suit == suit and not card.is_white_card:
               count += 1
        return count

class Trump:
    def __init__(self, card):
        self.card = card
        self.determined_suit = None

    @property
    def suit(self):
        if self.card.value == WIZARD:
            return self.determined_suit
        elif self.card.value == JESTER:
            return None
        else:
            return self.card.suit

    def __repr__(self):
        if self.determined_suit is None:
            return f"{self.card}"
        else:
            return f"{self.card} determined {self.determined_suit}"

class Play:
    def __init__(self, player_id, card):
        self.player_id = player_id
        self.card = card

class Trick:
    def __init__(self, trump: Trump, n_players: int):
        self.plays: List[Play] = []
        self.trump = trump
        self.n_players = n_players

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
            if play.card.value == WIZARD:
                return play
            if highest_card is None:
                highest_card = play.card
                winning_play = play
                continue
            if self.check_if_card_is_higher(highest_card, play.card):
                highest_card = play.card
                winning_play = play
        return winning_play

    @property
    def winner(self):
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
    def free_to_play(self): # returns true if all cards are allowed to be played
        return (self.first_to_act or self.first_card.value == WIZARD
                or all(play.card.value == JESTER for play in self.plays))

    def is_card_allowed_to_be_played(self, card):
        return card.value == WIZARD or card.value == JESTER or self.free_to_play or card.suit == self.first_card.suit

    def would_card_win_trick(self, card: Card) -> bool:
        return self.check_if_card_is_higher(self.winning_play.card, card)

    def check_if_card_is_higher(self, lower: Card, higher: Card) -> bool:
        if lower.value == WIZARD: return False
        if lower.value == JESTER and higher.value != JESTER:
            return True
        if ((higher.suit == lower.suit and higher.value > lower.value)
                or (higher.suit == self.trump.suit and lower.suit != self.trump.suit and higher.value != JESTER)):
            return True
        return False

    def __repr__(self):
        return ', '.join(map(str, (play.card for play in self.plays)))