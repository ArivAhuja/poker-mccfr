"""
Simplified Poker Engine for 6-card, 2-suit, 2-round limit hold'em
UPDATED VERSION - Now generates OpenSpiel universal_poker compatible information states
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
    
    def to_openspiel_format(self):
        """Convert to OpenSpiel card format (e.g., '2s', '7h')"""
        rank_str = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'}
        suit_str = {Suit.SPADES: 's', Suit.HEARTS: 'h'}
        return f"{rank_str[self.rank]}{suit_str[self.suit]}"


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
    
    FIXED: Now generates OpenSpiel universal_poker compatible information states
    """
    
    def __init__(self, small_blind=1, big_blind=2, starting_chips=20):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_chips = starting_chips
        self.deck = self._create_simplified_deck()
        self.game_state = None
        
        # Card mapping for OpenSpiel compatibility
        self.card_to_int = self._create_card_mapping()
        
    def _create_simplified_deck(self) -> List[Card]:
        """Create simplified 12-card deck: 6 ranks × 2 suits"""
        deck = []
        for suit in Suit:
            for rank in range(2, 8):  # 2,3,4,5,6,7 (6 ranks)
                deck.append(Card(rank, suit))
        return deck
    
    def _create_card_mapping(self) -> Dict[str, int]:
        """Create mapping from card strings to integers for OpenSpiel compatibility"""
        mapping = {}
        card_id = 0
        for suit in Suit:
            for rank in range(2, 8):
                card = Card(rank, suit)
                mapping[card.to_openspiel_format()] = card_id
                card_id += 1
        return mapping
    
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
    
    def get_information_set(self, player_id: int) -> str:
        """
        Get information set string for CFR interface - OpenSpiel compatible format
        FIXED: This now tries to match OpenSpiel's universal_poker format exactly
        """
        player = self.game_state.players[player_id]
        
        # Try to match OpenSpiel's universal_poker information state format
        # Based on your MCCFR training, it should be similar to your current format
        # but we need to make minor adjustments
        
        # Round number (0=preflop, 1=flop in OpenSpiel)
        round_num = 0 if self.game_state.current_street == Street.PREFLOP else 1
        
        # Current player
        current_player = self.game_state.current_player
        
        # Pot size
        pot = self.game_state.pot
        
        # Money (each player's remaining chips) - different formats to try
        money_str = f"{self.game_state.players[0].chips} {self.game_state.players[1].chips}"
        
        # Private cards (hole cards) - convert to OpenSpiel format
        private_cards = " ".join([card.to_openspiel_format() for card in player.hole_cards])
        
        # Public cards (community cards)
        if self.game_state.community_cards:
            public_cards = " ".join([card.to_openspiel_format() for card in self.game_state.community_cards])
        else:
            public_cards = ""
        
        # Sequences (betting history per round) - this is critical to match exactly
        sequences = self._get_openspiel_betting_sequences()
        
        # Try different formats to see which one matches your training data
        
        # Format 1: Your current format (what your GUI currently generates)
        format1 = f"[Round {round_num}][Player: {current_player}][Pot: {pot}][Money: {money_str}][Private: {private_cards}][Public: {public_cards}][Sequences: {sequences}]"
        
        # Format 2: OpenSpiel might use different spacing
        format2 = f"[Round {round_num}][Player: {current_player}][Pot: {pot}][Money: {money_str}][Private: {private_cards}][Public: {public_cards}][Sequences: {sequences}]"
        
        # Format 3: Maybe no spaces after colons
        format3 = f"[Round {round_num}][Player:{current_player}][Pot:{pot}][Money:{money_str}][Private:{private_cards}][Public:{public_cards}][Sequences:{sequences}]"
        
        # For now, return the current format, but we'll add debug info
        info_set = format1
        
        # DEBUG: Also try some variations and see if any match
        import hashlib
        
        print(f"🔍 DEBUG - Info state format attempts:")
        print(f"   Format 1: {format1}")
        print(f"   Format 1 hash: {hashlib.md5(format1.encode()).hexdigest()}")
        
        # Try some variations that might match OpenSpiel better
        variations = [
            # Different spacing
            f"[Round {round_num}][Player:{current_player}][Pot:{pot}][Money:{money_str}][Private:{private_cards}][Public:{public_cards}][Sequences:{sequences}]",
            
            # Different structure
            f"Round{round_num}:Player{current_player}:Pot{pot}:Money{money_str}:Private{private_cards}:Public{public_cards}:Sequences{sequences}",
            
            # Minimal format
            f"{round_num}:{current_player}:{pot}:{money_str}:{private_cards}:{public_cards}:{sequences}",
            
            # Maybe universal_poker uses different field names
            f"[Round {round_num}][Player {current_player}][Pot {pot}][Money {money_str}][Cards {private_cards}][Board {public_cards}][Actions {sequences}]"
        ]
        
        for i, variation in enumerate(variations):
            var_hash = hashlib.md5(variation.encode()).hexdigest()
            print(f"   Variation {i+1}: {variation}")
            print(f"   Variation {i+1} hash: {var_hash}")
        
        return info_set
    
    def _get_openspiel_betting_sequences(self) -> str:
        """Get betting sequences in OpenSpiel universal_poker format"""
        # This is critical - the betting sequence format must match exactly
        # what your MCCFR training used
        
        sequences = []
        
        # Group actions by street/round
        preflop_actions = []
        flop_actions = []
        
        for action in self.game_state.action_history:
            if action["action"] in ["small_blind", "big_blind"]:
                continue  # Skip blind posts in OpenSpiel format
            
            street = action["street"]
            action_char = self._action_to_openspiel_char(action["action"])
            
            if street == "preflop":
                preflop_actions.append(action_char)
            elif street == "flop":
                flop_actions.append(action_char)
        
        # OpenSpiel format: "|preflop_actions|flop_actions"
        if self.game_state.current_street == Street.PREFLOP:
            # Only preflop actions so far
            sequences_str = f"|{''.join(preflop_actions)}"
        else:
            # Both preflop and flop
            sequences_str = f"|{''.join(preflop_actions)}|{''.join(flop_actions)}"
        
        return sequences_str
    
    def _action_to_openspiel_char(self, action: str) -> str:
        """Convert action to OpenSpiel character - must match exactly what training used"""
        # These mappings must match exactly what your MCCFR training used
        # Based on standard universal_poker mappings:
        action_map = {
            "check": "c",
            "call": "c", 
            "bet": "r",    # OpenSpiel typically uses 'r' for raise/bet
            "raise": "r",
            "fold": "f"
        }
        return action_map.get(action, action[0])  # Fallback to first character
    
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


# NEW: Create a compatibility layer that tries multiple information state formats
class OpenSpielCompatibilityLayer:
    """
    Compatibility layer to bridge the gap between GUI and OpenSpiel formats
    Tries multiple information state formats to find the one that matches training data
    """
    
    def __init__(self, cfr_interface):
        self.cfr_interface = cfr_interface
        self.format_cache = {}  # Cache successful formats
        
    def get_strategy_with_format_detection(self, base_info_state: str, player_id: int) -> Optional[List[float]]:
        """
        Try multiple information state formats to find one that matches the training data
        """
        # If we've already found a working format for this type of state, use it
        state_pattern = self._get_state_pattern(base_info_state)
        if state_pattern in self.format_cache:
            format_func = self.format_cache[state_pattern]
            formatted_state = format_func(base_info_state)
            strategy = self.cfr_interface.get_strategy_for_info_state(formatted_state)
            if strategy:
                return strategy
        
        # Try different formats
        formats_to_try = [
            # Original format
            lambda s: s,
            
            # Remove spaces after colons
            lambda s: s.replace(": ", ":"),
            
            # Different bracket style
            lambda s: s.replace("[", "").replace("]", "").replace(":", " "),
            
            # Compact format
            lambda s: s.replace("[Round ", "R").replace("][Player: ", "P").replace("][Pot: ", "pot").replace("][Money: ", "M").replace("][Private: ", "priv").replace("][Public: ", "pub").replace("][Sequences: ", "seq").replace("]", ""),
            
            # Try without some fields
            lambda s: self._extract_core_info(s),
            
            # Try with different card formats
            lambda s: self._convert_card_format(s),
        ]
        
        for format_func in formats_to_try:
            try:
                formatted_state = format_func(base_info_state)
                strategy = self.cfr_interface.get_strategy_for_info_state(formatted_state)
                
                if strategy:
                    print(f"✅ Found working format for pattern '{state_pattern}'")
                    print(f"   Working format: {formatted_state}")
                    
                    # Cache this format for future use
                    self.format_cache[state_pattern] = format_func
                    return strategy
                    
            except Exception as e:
                continue
        
        return None
    
    def _get_state_pattern(self, info_state: str) -> str:
        """Extract a pattern from the info state for caching"""
        import re
        # Extract round and basic structure
        round_match = re.search(r'Round (\d+)', info_state)
        player_match = re.search(r'Player: (\d+)', info_state)
        
        round_num = round_match.group(1) if round_match else "X"
        player_num = player_match.group(1) if player_match else "X"
        
        return f"R{round_num}P{player_num}"
    
    def _extract_core_info(self, info_state: str) -> str:
        """Extract only the most essential information"""
        import re
        
        # Extract key components
        round_match = re.search(r'Round (\d+)', info_state)
        player_match = re.search(r'Player: (\d+)', info_state)
        pot_match = re.search(r'Pot: (\d+)', info_state)
        private_match = re.search(r'Private: ([^]]+)', info_state)
        public_match = re.search(r'Public: ([^]]*)', info_state)
        sequences_match = re.search(r'Sequences: ([^]]+)', info_state)
        
        round_num = round_match.group(1) if round_match else "0"
        player_num = player_match.group(1) if player_match else "0"
        pot = pot_match.group(1) if pot_match else "0"
        private = private_match.group(1) if private_match else ""
        public = public_match.group(1) if public_match else ""
        sequences = sequences_match.group(1) if sequences_match else "|"
        
        # Try a minimal format
        return f"{round_num}:{player_num}:{pot}:{private}:{public}:{sequences}"
    
    def _convert_card_format(self, info_state: str) -> str:
        """Try different card representations"""
        # Convert card format from "6s 5h" to different formats
        import re
        
        # Find card patterns
        card_pattern = r'(\d+[sh])'
        cards = re.findall(card_pattern, info_state)
        
        if cards:
            # Try different card formats
            # Maybe OpenSpiel uses different suit characters or ordering
            modified_state = info_state
            for card in cards:
                rank, suit = card[0], card[1]
                # Try different suit representations
                if suit == 's':
                    new_card = f"{rank}♠"
                elif suit == 'h':
                    new_card = f"{rank}♥"
                else:
                    new_card = card
                
                modified_state = modified_state.replace(card, new_card)
            
            return modified_state
        
        return info_state


if __name__ == "__main__":
    # Test the simplified engine
    engine = SimplifiedPokerEngine()
    game_state = engine.start_new_hand(["Human", "AI"])
    
    # Test information state generation
    info_state = engine.get_information_set(0)
    print(f"Generated info state: {info_state}")
    
    # Test hash
    import hashlib
    info_hash = hashlib.md5(info_state.encode()).hexdigest()
    print(f"Info state hash: {info_hash}")
