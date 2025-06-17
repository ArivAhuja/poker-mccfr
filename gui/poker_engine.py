"""
Simplified Poker Engine for 6-card, 2-suit, 2-round limit hold'em
UPDATED VERSION - Now generates OpenSpiel-compatible information states
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, asdict


class Suit(Enum):
    SPADES = "♠"
    HEARTS = "♥"


class Action(Enum):
    FOLD = "fold"
    CALL = "call"
    CHECK = "check"
    BET = "bet"
    RAISE = "raise"


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"


@dataclass
class Card:
    rank: int  # 2-7 (only 6 ranks: 2,3,4,5,6,7)
    suit: Suit
    
    def __str__(self):
        rank_str = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
        return f"{rank_str[self.rank]}{self.suit.value}"
    
    def to_dict(self):
        return {"rank": self.rank, "suit": self.suit.value}


@dataclass 
class Player:
    id: int
    name: str
    chips: int
    hole_cards: List[Card]
    current_bet: int = 0
    total_bet: int = 0
    folded: bool = False
    all_in: bool = False
    has_acted: bool = False
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name, 
            "chips": self.chips,
            "hole_cards": [card.to_dict() for card in self.hole_cards],
            "current_bet": self.current_bet,
            "total_bet": self.total_bet,
            "folded": self.folded,
            "all_in": self.all_in
        }


@dataclass
class GameState:
    players: List[Player]
    community_cards: List[Card]
    pot: int
    current_street: Street
    current_player: int
    dealer_button: int
    small_blind: int
    big_blind: int
    action_history: List[Dict]
    game_over: bool = False
    winner: Optional[int] = None
    raise_count: int = 0  # Track raises for max raises limit
    
    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "community_cards": [c.to_dict() for c in self.community_cards],
            "pot": self.pot,
            "current_street": self.current_street.value,
            "current_player": self.current_player,
            "dealer_button": self.dealer_button,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "action_history": self.action_history,
            "game_over": self.game_over,
            "winner": self.winner,
            "raise_count": self.raise_count
        }


class SimplifiedPokerEngine:
    """
    Simplified Poker Engine matching your ACPC GAMEDEF:
    - 2 players
    - 2 rounds (preflop + flop only)  
    - 6 ranks (2,3,4,5,6,7) × 2 suits = 12 cards total
    - Limit betting: 2-4 structure
    - Max 2 raises per round
    - Starting stack: 20 chips
    """
    
    def __init__(self, small_blind=1, big_blind=2, starting_chips=20):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_chips = starting_chips
        self.deck = self._create_simplified_deck()
        self.game_state = None
        
    def _create_simplified_deck(self) -> List[Card]:
        """Create simplified 12-card deck: 6 ranks × 2 suits"""
        deck = []
        for suit in Suit:
            for rank in range(2, 8):  # 2,3,4,5,6,7 (6 ranks)
                deck.append(Card(rank, suit))
        return deck
    
    def _shuffle_deck(self):
        """Shuffle the deck"""
        random.shuffle(self.deck)
    
    def start_new_hand(self, player_names: List[str] = None) -> GameState:
        """Start a new hand"""
        if player_names is None:
            player_names = ["Human", "AI"]
        
        # Reset deck
        self.deck = self._create_simplified_deck()
        self._shuffle_deck()
        
        # Create players - keep existing chip counts if players exist
        players = []
        if self.game_state and len(self.game_state.players) == len(player_names):
            # Keep existing chip counts
            for i, name in enumerate(player_names):
                existing_chips = self.game_state.players[i].chips
                players.append(Player(
                    id=i,
                    name=name,
                    chips=existing_chips,
                    hole_cards=[],
                    current_bet=0,
                    total_bet=0,
                    folded=False,
                    all_in=False,
                    has_acted=False
                ))
        else:
            # New game - fresh chips
            for i, name in enumerate(player_names):
                players.append(Player(
                    id=i,
                    name=name,
                    chips=self.starting_chips,
                    hole_cards=[],
                    current_bet=0,
                    total_bet=0,
                    folded=False,
                    all_in=False,
                    has_acted=False
                ))
        
        # Deal hole cards (2 cards each)
        for _ in range(2):
            for player in players:
                if len(self.deck) > 0:
                    player.hole_cards.append(self.deck.pop())
        
        # FIXED: Rotate dealer button between hands
        if self.game_state:
            # Alternate button: 0 -> 1 -> 0 -> 1...
            new_button = (self.game_state.dealer_button + 1) % 2
        else:
            # First hand - start with player 0 as button
            new_button = 0
        
        # Initialize game state
        self.game_state = GameState(
            players=players,
            community_cards=[],
            pot=0,
            current_street=Street.PREFLOP,
            current_player=0,
            dealer_button=new_button,  # Use rotated button
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            action_history=[],
            raise_count=0
        )
        
        # Post blinds
        self._post_blinds()
        
        return self.game_state
    
    def _post_blinds(self):
        """Post small and big blinds (FIXED: correct heads-up positioning)"""
        # FIXED: In heads-up poker, button is BIG BLIND, other player is SMALL BLIND
        # This is opposite of full ring poker
        bb_player = self.game_state.dealer_button  # Button = big blind in heads-up
        sb_player = (self.game_state.dealer_button + 1) % 2  # Other = small blind
        
        # Small blind
        self.game_state.players[sb_player].current_bet = self.small_blind
        self.game_state.players[sb_player].chips -= self.small_blind
        self.game_state.players[sb_player].total_bet += self.small_blind
        self.game_state.pot += self.small_blind
        
        # Big blind  
        self.game_state.players[bb_player].current_bet = self.big_blind
        self.game_state.players[bb_player].chips -= self.big_blind
        self.game_state.players[bb_player].total_bet += self.big_blind
        self.game_state.pot += self.big_blind
        
        # Record blind actions
        self.game_state.action_history.append({
            "player": sb_player,
            "action": "small_blind",
            "amount": self.small_blind,
            "street": self.game_state.current_street.value
        })
        
        self.game_state.action_history.append({
            "player": bb_player,
            "action": "big_blind", 
            "amount": self.big_blind,
            "street": self.game_state.current_street.value
        })
        
        # FIXED: Small blind acts first preflop in heads-up
        self.game_state.current_player = sb_player
        
        print(f"🎰 New hand - Button: Player {self.game_state.dealer_button}")
        print(f"   Small Blind: Player {sb_player} (${self.small_blind})")
        print(f"   Big Blind: Player {bb_player} (${self.big_blind})")
        print(f"   First to act: Player {sb_player}")
    
    def get_valid_actions(self) -> List[Action]:
        """Get valid actions for current player - FIXED VERSION"""
        if self.game_state.game_over:
            return []
        
        player = self.game_state.players[self.game_state.current_player]
        if player.folded or player.all_in:
            return []
        
        actions = []
        
        # Calculate call amount
        max_bet = max(p.current_bet for p in self.game_state.players if not p.folded)
        call_amount = max_bet - player.current_bet
        
        # Always can fold (except in some check situations)
        actions.append(Action.FOLD)
        
        # FIXED: Better action logic based on game state
        if call_amount == 0:
            # No bet to call - can check
            actions.append(Action.CHECK)
        else:
            # There's a bet to call
            if player.chips >= call_amount:
                actions.append(Action.CALL)
        
        # Can bet/raise if have chips AND haven't hit max raises
        if player.chips >= self.big_blind and self.game_state.raise_count < 2:
            if call_amount == 0:
                actions.append(Action.BET)
            else:
                actions.append(Action.RAISE)
        
        return actions
    
    def make_action(self, action: Action, amount: int = 0) -> bool:
        """Make an action for the current player - FIXED VERSION"""
        if self.game_state.game_over:
            return False
        
        player = self.game_state.players[self.game_state.current_player]
        valid_actions = self.get_valid_actions()
        
        if action not in valid_actions:
            print(f"❌ Invalid action {action.value}, valid: {[a.value for a in valid_actions]}")
            return False
        
        print(f"🎲 Player {player.id} ({player.name}) chooses {action.value}")
        
        # Record action
        action_record = {
            "player": player.id,
            "action": action.value,
            "amount": amount,
            "street": self.game_state.current_street.value
        }
        self.game_state.action_history.append(action_record)
        
        # Mark player as having acted
        player.has_acted = True
        
        # Execute action
        if action == Action.FOLD:
            player.folded = True
            # Check if game ends due to fold
            active_players = [p for p in self.game_state.players if not p.folded]
            if len(active_players) == 1:
                self._end_hand_by_fold()
                return True
        
        elif action == Action.CHECK:
            pass  # No chips change
        
        elif action == Action.CALL:
            max_bet = max(p.current_bet for p in self.game_state.players if not p.folded)
            call_amount = max_bet - player.current_bet
            call_amount = min(call_amount, player.chips)
            
            player.current_bet += call_amount
            player.chips -= call_amount
            player.total_bet += call_amount
            self.game_state.pot += call_amount
            
            if player.chips == 0:
                player.all_in = True
        
        elif action in [Action.BET, Action.RAISE]:
            # Fixed limit betting: bet size is always big blind (2)
            if action == Action.RAISE:
                # Raise = call + bet
                max_bet = max(p.current_bet for p in self.game_state.players if not p.folded)
                call_amount = max_bet - player.current_bet
                total_bet_amount = call_amount + self.big_blind
            else:  # BET
                total_bet_amount = self.big_blind
            
            # Ensure we don't bet more than we have
            total_bet_amount = min(total_bet_amount, player.chips)
            
            player.current_bet += total_bet_amount
            player.chips -= total_bet_amount
            player.total_bet += total_bet_amount
            self.game_state.pot += total_bet_amount
            
            # Increment raise count
            self.game_state.raise_count += 1
            
            if player.chips == 0:
                player.all_in = True
            
            # Reset has_acted for all other players when there's a bet/raise
            for p in self.game_state.players:
                if p.id != player.id and not p.folded and not p.all_in:
                    p.has_acted = False
        
        # Check if betting round is over after this action
        if self._is_betting_round_over():
            self._advance_street()
        else:
            self._advance_player()
        
        return True
    
    def _end_hand_by_fold(self):
        """End hand when someone folds"""
        active_players = [p for p in self.game_state.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.game_state.pot
            self.game_state.winner = winner.id
            self.game_state.game_over = True
            print(f"🏆 Player {winner.id} ({winner.name}) wins by fold!")
    
    def _is_betting_round_over(self) -> bool:
        """FIXED: Check if current betting round is over"""
        active_players = [p for p in self.game_state.players if not p.folded and not p.all_in]
        
        # If only one or no active players, round is over
        if len(active_players) <= 1:
            return True
        
        # Get all current bets from active players
        active_bets = [p.current_bet for p in active_players]
        
        # Check if all active players have the same bet amount
        equal_bets = len(set(active_bets)) <= 1
        
        # Check if all active players have acted at least once this round
        all_acted = all(p.has_acted for p in active_players)
        
        print(f"🔍 Betting round check: equal_bets={equal_bets}, all_acted={all_acted}")
        print(f"   Active bets: {active_bets}")
        print(f"   Has acted: {[p.has_acted for p in active_players]}")
        
        # FIXED: Special case for preflop with big blind option
        if self.game_state.current_street == Street.PREFLOP:
            bb_player = self.game_state.dealer_button  # Button is BB in heads-up
            bb_player_obj = self.game_state.players[bb_player]
            
            # If bets are equal but BB hasn't had a chance to act after a call, give BB option
            if equal_bets and not bb_player_obj.folded and not bb_player_obj.all_in:
                # Check if there was any betting action (not just blinds)
                non_blind_actions = [a for a in self.game_state.action_history 
                                   if a["street"] == "preflop" and 
                                   a["action"] not in ["small_blind", "big_blind"]]
                
                if non_blind_actions and not bb_player_obj.has_acted:
                    print(f"   BB ({bb_player}) gets option to act")
                    return False
        
        # Round is over if all players have acted and bets are equal
        return all_acted and equal_bets
    
    def _advance_player(self):
        """Move to next active player"""
        original_player = self.game_state.current_player
        
        # Find the next active player
        for _ in range(len(self.game_state.players)):
            self.game_state.current_player = (self.game_state.current_player + 1) % len(self.game_state.players)
            player = self.game_state.players[self.game_state.current_player]
            if not player.folded and not player.all_in:
                break
        
        print(f"👉 Turn advances: Player {original_player} → Player {self.game_state.current_player}")
    
    def _advance_street(self):
        """Advance to next street"""
        print(f"🏁 Advancing from {self.game_state.current_street.value}")
        
        # Reset current bets, has_acted flags, and raise count for new street
        for player in self.game_state.players:
            player.current_bet = 0
            player.has_acted = False
        self.game_state.raise_count = 0
        
        # Deal community cards based on current street
        if self.game_state.current_street == Street.PREFLOP:
            # Deal flop (3 cards) - only street transition in simplified game
            for _ in range(3):
                if len(self.deck) > 0:
                    self.game_state.community_cards.append(self.deck.pop())
            self.game_state.current_street = Street.FLOP
            print(f"🃏 Dealt flop: {[str(c) for c in self.game_state.community_cards]}")
        else:
            # After flop, go to showdown (only 2 rounds total)
            self._showdown()
            return
        
        # FIXED: Set first player to act post-flop (small blind acts first)
        # In heads-up, the non-button player is small blind and acts first post-flop
        sb_player = (self.game_state.dealer_button + 1) % 2
        self.game_state.current_player = sb_player
        
        # Find first active player if SB is folded/all-in
        for _ in range(len(self.game_state.players)):
            player = self.game_state.players[self.game_state.current_player]
            if not player.folded and not player.all_in:
                break
            self.game_state.current_player = (self.game_state.current_player + 1) % len(self.game_state.players)
        
        print(f"🎯 New street: {self.game_state.current_street.value}, current player: {self.game_state.current_player}")
    
    def _showdown(self):
        """Determine winner and distribute pot"""
        active_players = [p for p in self.game_state.players if not p.folded]
        
        if len(active_players) == 1:
            # Only one player left
            winner = active_players[0]
            winner.chips += self.game_state.pot
            self.game_state.winner = winner.id
        else:
            # Evaluate hands for showdown
            player_hands = []
            for player in active_players:
                hand_strength = SimplifiedHandEvaluator.evaluate_hand(
                    player.hole_cards, 
                    self.game_state.community_cards
                )
                player_hands.append((player, hand_strength))
                print(f"Player {player.id} ({player.name}) hand strength: {hand_strength}")
            
            # Find the best hand(s)
            best_strength = max(hand_strength for _, hand_strength in player_hands)
            winners = [player for player, strength in player_hands if strength == best_strength]
            
            print(f"Best strength: {best_strength}, winners: {[w.name for w in winners]}")
            
            # Distribute pot among winners
            pot_share = self.game_state.pot // len(winners)
            for winner in winners:
                winner.chips += pot_share
            
            # Set winner (if tie, just pick first winner for display)
            self.game_state.winner = winners[0].id
        
        self.game_state.game_over = True

    _R = "234567"
    _S = {"♠": "s", "♥": "h"}

    def _card(self, c: Card) -> str:
        return f"{self._R[c.rank-2]}{self._S[c.suit.value]}"

    _A = {"check": "c", "call": "c", "bet": "r", "raise": "r", "fold": "f"}

    def _hist(self, rec: dict) -> str:
        return self._A.get(rec["action"], "")
    
    # ---------- helpers ----------
    _RANK_CHAR = "234567"
    _SUIT_CHAR = {"♠": "s",   # suit-id 0
                "♥": "h"}   # suit-id 1
    _ACTION_CHR = {"check": "c", "call": "c",
                   "bet": "r",   "raise": "r",
                   "fold": "f"}

    def _card_str(self, c: Card) -> str:
        return f"{self._RANK_CHAR[c.rank-2]}{self._SUIT_CHAR[c.suit.value]}"

    def _sorted_cards(self, cards):
        # ascending rank, then suit (“c” < “d”)
        return "".join(
            self._card_str(c)
            for c in sorted(cards, key=lambda c: (c.rank, self._SUIT_CHAR[c.suit.value]))
        )
    def _hist(self):
        return "".join(self._ACTION_CHR[a["action"]]
                       for a in self.game_state.action_history
                       if a["action"] not in ("small_blind", "big_blind"))

    # ---------- canonical info-state ----------
    def get_information_set(self, pid: int) -> str:
        pl     = self.game_state.players[pid]
        hole   = self._sorted_cards(pl.hole_cards)               # e.g. "6h3s"
        board  = self._sorted_cards(self.game_state.community_cards)
        hist   = self._hist()                                    # e.g. "cr"

        if board:            # after flop
            return f"{hole}/{board}|{hist}"
        else:                # pre-flop
            return f"{hole}|{hist}"


    def _cards_to_openspiel_format(self, cards: List[Card]) -> str:
        """Convert cards to OpenSpiel format"""
        if not cards:
            return ""
        
        openspiel_cards = []
        for card in cards:
            # Convert rank number to OpenSpiel format
            rank_str = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}[card.rank]
            
            # Convert suit to OpenSpiel format (check if they use different suit chars)
            suit_str = {'♠': 's', '♥': 'h'}[card.suit.value]  # Spades=s, Hearts=h
            
            openspiel_cards.append(f"{rank_str}{suit_str}")
        
        return " ".join(openspiel_cards)
    
    def _get_betting_sequences(self) -> str:
        """Get betting sequences in OpenSpiel format"""
        # OpenSpiel uses format like "|c|" or "|cr|" for betting sequences per round
        # Each round separated by |
        
        sequences = []
        
        # Group actions by street/round
        preflop_actions = []
        flop_actions = []
        
        for action in self.game_state.action_history:
            if action["action"] in ["small_blind", "big_blind"]:
                continue  # Skip blind posts
            
            street = action["street"]
            action_char = self._action_to_openspiel_char(action["action"])
            
            if street == "preflop":
                preflop_actions.append(action_char)
            elif street == "flop":
                flop_actions.append(action_char)
        
        # Format: "|preflop_actions|flop_actions"
        if self.game_state.current_street == Street.PREFLOP:
            # Only preflop, but still need the separator
            sequences_str = f"|{''.join(preflop_actions)}"
        else:
            # Both preflop and flop
            sequences_str = f"|{''.join(preflop_actions)}|{''.join(flop_actions)}"
        
        return sequences_str
    
    def _action_to_openspiel_char(self, action: str) -> str:
        """Convert action to OpenSpiel character"""
        # Based on OpenSpiel source, common mappings:
        action_map = {
            "check": "c",
            "call": "c", 
            "bet": "r",    # OpenSpiel uses 'r' for raise/bet
            "raise": "r",
            "fold": "f"
        }
        return action_map.get(action, action[0])  # Fallback to first character
    
    def _card_to_string(self, card: Card) -> str:
        """Convert card to string format for info sets (legacy method)"""
        rank_map = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
        suit_map = {Suit.SPADES: 's', Suit.HEARTS: 'h'}
        return f"{rank_map[card.rank]}{suit_map[card.suit]}"
    
    def get_game_state_dict(self) -> Dict:
        """Get game state as dictionary for JSON serialization"""
        return self.game_state.to_dict() if self.game_state else None


class SimplifiedHandEvaluator:
    """
    Hand evaluation for simplified 6-card, 2-suit poker
    """
    
    @staticmethod
    def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> int:
        """
        Returns hand strength (higher = better)
        Simplified for 6 ranks (2,3,4,5,6,7) and 2 suits
        """
        all_cards = hole_cards + community_cards
        
        if len(all_cards) < 2:
            return 0
        
        # Count ranks and suits
        rank_counts = {}
        suit_counts = {}
        ranks = []
        
        for card in all_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
            ranks.append(card.rank)
        
        ranks.sort(reverse=True)
        
        # Check for flush (5+ cards of same suit)
        has_flush = any(count >= 5 for count in suit_counts.values()) if len(all_cards) >= 5 else False
        
        # Check for straight (simplified for 6 ranks)
        unique_ranks = sorted(set(ranks), reverse=True)
        has_straight = SimplifiedHandEvaluator._check_straight(unique_ranks)
        
        # Count pairs, trips, quads
        counts = sorted(rank_counts.values(), reverse=True)
        rank_values = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True)
        
        # Hand rankings (higher = better) - simplified for 6-card game
        if has_straight and has_flush:
            return 8000000 + max(unique_ranks)  # Straight flush
        elif len(counts) > 0 and counts[0] == 4:
            return 7000000 + rank_values[0] * 1000 + (rank_values[1] if len(rank_values) > 1 else 0)  # Four of a kind
        elif len(counts) > 1 and counts[0] == 3 and counts[1] >= 2:
            return 6000000 + rank_values[0] * 1000 + rank_values[1]  # Full house
        elif has_flush:
            return 5000000 + sum(rank * (10 ** i) for i, rank in enumerate(unique_ranks[:5]))  # Flush
        elif has_straight:
            return 4000000 + max(unique_ranks)  # Straight
        elif len(counts) > 0 and counts[0] == 3:
            kickers = rank_values[1:3] if len(rank_values) > 1 else [0]
            return 3000000 + rank_values[0] * 10000 + sum(k * (10 ** i) for i, k in enumerate(kickers))  # Three of a kind
        elif len(counts) > 1 and counts[0] == 2 and counts[1] == 2:
            pairs = rank_values[:2]
            kicker = rank_values[2] if len(rank_values) > 2 else 0
            return 2000000 + pairs[0] * 10000 + pairs[1] * 100 + kicker  # Two pair
        elif len(counts) > 0 and counts[0] == 2:
            kickers = rank_values[1:4] if len(rank_values) > 1 else [0]
            return 1000000 + rank_values[0] * 100000 + sum(k * (10 ** i) for i, k in enumerate(kickers))  # One pair
        else:
            # High card - use available cards
            top_cards = unique_ranks[:min(5, len(unique_ranks))]
            return sum(rank * (10 ** i) for i, rank in enumerate(top_cards))  # High card
    
    @staticmethod
    def _check_straight(ranks):
        """Check if ranks contain a straight (simplified for 6 ranks: 2,3,4,5,6,7)"""
        if len(ranks) < 5:
            return False
        
        # Only possible straights with ranks 2,3,4,5,6,7:
        # 7-6-5-4-3 and 6-5-4-3-2
        possible_straights = [
            [7, 6, 5, 4, 3],
            [6, 5, 4, 3, 2]
        ]
        
        for straight in possible_straights:
            if all(rank in ranks for rank in straight):
                return True
        
        return False


if __name__ == "__main__":
    # Test the simplified engine
    engine = SimplifiedPokerEngine()
    game_state = engine.start_new_hand(["Human", "AI"])
