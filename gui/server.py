"""
Flask server for simplified poker GUI
FIXED VERSION - Properly handles AI turns automatically
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
import os
from pathlib import Path
from typing import Dict, Any

from poker_engine import SimplifiedPokerEngine, Action, SimplifiedHandEvaluator
from cfr_interface import AIPlayer, load_cfr_model

# Get the directory where this script is located
current_dir = Path(__file__).parent
template_dir = current_dir / 'templates'  # Go up one level to find templates

# Create Flask app with explicit template directory
app = Flask(__name__, template_folder=str(template_dir))
CORS(app)  # Enable CORS for frontend-backend communication

# Global game state
game_engine = None
ai_player = None
game_session = {}


@app.route('/')
def index():
    """Serve the main poker interface"""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new poker game"""
    global game_engine, ai_player, game_session
    
    try:
        data = request.get_json() or {}
        
        # Initialize simplified game engine
        small_blind = data.get('small_blind', 1)
        big_blind = data.get('big_blind', 2)
        starting_chips = data.get('starting_chips', 20)  # Simplified game default
        
        game_engine = SimplifiedPokerEngine(small_blind, big_blind, starting_chips)
        
        # Initialize AI player with trained CFR model
        strategy_file = data.get('strategy_file', None)
        try:
            ai_player = load_cfr_model(strategy_file)
            print(f"🤖 AI Player initialized - Model loaded: {ai_player.is_model_loaded()}")
        except Exception as e:
            print(f"⚠️  Error loading CFR model: {e}")
            ai_player = AIPlayer("AI")  # Fallback
        
        # Start new hand
        player_names = ["You", "AI"]
        game_state = game_engine.start_new_hand(player_names)
        
        # Initialize session
        game_session = {
            'game_id': int(time.time()),
            'hand_count': 1,
            'player_wins': 0,
            'ai_wins': 0,
            'model_loaded': ai_player.is_model_loaded()
        }
        
        # FIXED: Check if AI should act first and handle it automatically
        if game_state.current_player == 1 and not game_state.game_over:  # AI is player 1
            return _handle_ai_turn_with_initial_state()
        
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'session': game_session,
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
            'message': f'New simplified game started! {"CFR model loaded" if ai_player.is_model_loaded() else "Using fallback AI"} - Your turn.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to start new game'
        }), 500


def _handle_ai_turn_with_initial_state():
    """Handle AI's turn at game start and return proper response"""
    global game_engine, ai_player, game_session
    
    try:
        game_state = game_engine.game_state
        
        # Keep making AI actions until it's human's turn or game ends
        ai_actions_taken = []
        
        while game_state.current_player == 1 and not game_state.game_over:
            valid_actions = game_engine.get_valid_actions()
            
            if not valid_actions:
                break
            
            print(f"🤖 AI thinking... Valid actions: {[a.value for a in valid_actions]}")
            
            # Get AI action using trained CFR model
            ai_action = ai_player.get_action(game_state, valid_actions)
            
            if ai_action:
                print(f"🤖 AI chooses: {ai_action.value}")
                ai_actions_taken.append(ai_action.value)
                
                # Make AI action
                success = game_engine.make_action(ai_action)
                
                if not success:
                    print(f"❌ AI action {ai_action.value} failed")
                    break
            else:
                print("⚠️  AI returned no action, using fallback")
                ai_action = valid_actions[0]  # Fallback
                ai_actions_taken.append(ai_action.value)
                game_engine.make_action(ai_action)
            
            # Update game state after AI action
            game_state = game_engine.game_state
        
        # Check if game is over after AI actions
        if game_state.game_over:
            return _handle_game_over()
        
        # Get AI strategy info for display
        ai_strategy = ai_player.get_strategy_info(game_state, 1)
        
        # Create message about AI actions
        if len(ai_actions_taken) == 1:
            ai_message = f"AI {ai_actions_taken[0]}s."
        elif len(ai_actions_taken) > 1:
            ai_message = f"AI actions: {', '.join(ai_actions_taken)}."
        else:
            ai_message = "AI had no valid actions."
        
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'session': game_session,
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
            'ai_actions': ai_actions_taken,
            'ai_strategy': ai_strategy,
            'message': f'{ai_message} Your turn.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in AI turn: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error in AI turn'
        }), 500


@app.route('/api/make_action', methods=['POST'])
def make_action():
    """Make a player action"""
    global game_engine, ai_player
    
    if not game_engine:
        return jsonify({
            'success': False,
            'error': 'No active game',
            'message': 'Please start a new game first'
        }), 400
    
    try:
        data = request.get_json()
        action_str = data.get('action')
        amount = data.get('amount', 0)
        
        # FIXED: Ensure it's actually the human's turn
        if game_engine.game_state.current_player != 0:
            return jsonify({
                'success': False,
                'error': 'Not your turn',
                'message': f'It is currently Player {game_engine.game_state.current_player}\'s turn'
            }), 400
        
        # Convert string to Action enum
        try:
            action = Action(action_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Invalid action: {action_str}',
                'message': 'Invalid action specified'
            }), 400
        
        # Make player action
        success = game_engine.make_action(action, amount)
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Invalid action',
                'message': 'Action not allowed in current game state'
            }), 400
        
        game_state = game_engine.game_state
        
        # Check if game is over
        if game_state.game_over:
            return _handle_game_over()
        
        # FIXED: Check if it's AI's turn and handle all AI actions
        if game_state.current_player == 1 and not game_state.game_over:  # AI is player 1
            return _handle_ai_turn()
        
        # Return updated game state for human player
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
            'message': 'Your turn.' if game_state.current_player == 0 else 'Waiting for AI...'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing action'
        }), 500


def _handle_ai_turn():
    """Handle AI's turn with CFR model - can handle multiple AI actions in sequence"""
    global game_engine, ai_player
    
    try:
        game_state = game_engine.game_state
        ai_actions_taken = []
        
        # Keep making AI actions until it's human's turn or game ends
        while game_state.current_player == 1 and not game_state.game_over:
            valid_actions = game_engine.get_valid_actions()
            
            if not valid_actions:
                print("🤖 AI has no valid actions")
                break
            
            print(f"🤖 AI thinking... Valid actions: {[a.value for a in valid_actions]}")
            
            # Get AI action using trained CFR model
            ai_action = ai_player.get_action(game_state, valid_actions)
            
            if ai_action:
                print(f"🤖 AI chooses: {ai_action.value}")
                ai_actions_taken.append(ai_action.value)
                
                # Make AI action
                success = game_engine.make_action(ai_action)
                
                if not success:
                    return jsonify({
                        'success': False,
                        'error': 'AI action failed',
                        'message': f'AI made an invalid action: {ai_action.value}'
                    }), 500
            else:
                print("⚠️  AI returned no action, using fallback")
                ai_action = valid_actions[0]  # Fallback
                ai_actions_taken.append(ai_action.value)
                game_engine.make_action(ai_action)
            
            # Update game state after each AI action
            game_state = game_engine.game_state
        
        # Check if game is over after AI actions
        if game_state.game_over:
            return _handle_game_over()
        
        # Get AI strategy info for display
        ai_strategy = ai_player.get_strategy_info(game_state, 1)
        
        # Create message about AI actions
        if len(ai_actions_taken) == 1:
            ai_message = f"AI {ai_actions_taken[0]}s."
        elif len(ai_actions_taken) > 1:
            ai_message = f"AI actions: {', '.join(ai_actions_taken)}."
        else:
            ai_message = "AI had no actions."
        
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
            'ai_actions': ai_actions_taken,
            'ai_strategy': ai_strategy,
            'message': f'{ai_message} Your turn.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in AI turn: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error in AI turn'
        }), 500


def _handle_game_over():
    """Handle end of game"""
    global game_session
    
    game_state = game_engine.game_state
    
    # Determine winner and update session stats
    winner_message = ""
    
    if game_state.winner == 0:  # Human wins
        game_session['player_wins'] += 1
        winner_message = "🎉 You win this hand!"
    elif game_state.winner == 1:  # AI wins
        game_session['ai_wins'] += 1
        winner_message = "🤖 AI wins this hand!"
    else:
        winner_message = "🤝 Hand ended in a tie!"
    
    # Show hand strengths for debugging
    if len(game_engine.game_state.players) == 2:
        human_player = game_engine.game_state.players[0]
        ai_player_state = game_engine.game_state.players[1]
        
        if not human_player.folded and not ai_player_state.folded:
            human_strength = SimplifiedHandEvaluator.evaluate_hand(
                human_player.hole_cards, 
                game_engine.game_state.community_cards
            )
            ai_strength = SimplifiedHandEvaluator.evaluate_hand(
                ai_player_state.hole_cards, 
                game_engine.game_state.community_cards
            )
            winner_message += f" (Your hand: {human_strength}, AI hand: {ai_strength})"
    
    response_data = {
        'success': True,
        'game_state': game_state.to_dict(),
        'session': game_session,
        'game_over': True,
        'message': winner_message
    }
    
    return jsonify(response_data)


@app.route('/api/new_hand', methods=['POST'])
def new_hand():
    """Start a new hand (same game, new cards)"""
    global game_engine, game_session
    
    if not game_engine:
        return jsonify({
            'success': False,
            'error': 'No active game',
            'message': 'Please start a new game first'
        }), 400
    
    try:
        # Start new hand
        game_state = game_engine.start_new_hand(["You", "AI"])
        
        # Update session
        game_session['hand_count'] += 1
        
        # FIXED: Check if AI should act first in new hand
        if game_state.current_player == 1 and not game_state.game_over:  # AI is player 1
            return _handle_ai_turn_with_initial_state()
        
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'session': game_session,
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
            'message': 'New hand dealt! Your turn.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to start new hand'
        }), 500


@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    """Get current game state"""
    if not game_engine or not game_engine.game_state:
        return jsonify({
            'success': False,
            'error': 'No active game',
            'message': 'No game in progress'
        }), 400
    
    try:
        game_state = game_engine.game_state
        
        response_data = {
            'success': True,
            'game_state': game_state.to_dict(),
            'session': game_session,
            'valid_actions': [action.value for action in game_engine.get_valid_actions()],
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error getting game state'
        }), 500


@app.route('/api/ai_strategy', methods=['GET'])
def get_ai_strategy():
    """Get AI strategy information for display"""
    if not game_engine or not ai_player:
        return jsonify({
            'success': False,
            'error': 'No active game'
        }), 400
    
    try:
        game_state = game_engine.game_state
        ai_strategy = ai_player.get_strategy_info(game_state, 1)
        
        return jsonify({
            'success': True,
            'ai_strategy': ai_strategy
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/load_strategy', methods=['POST'])
def load_strategy():
    """Load a different CFR strategy file"""
    global ai_player
    
    try:
        data = request.get_json()
        strategy_file = data.get('strategy_file')
        
        if not strategy_file or not os.path.exists(strategy_file):
            return jsonify({
                'success': False,
                'error': 'Strategy file not found',
                'message': f'File not found: {strategy_file}'
            }), 400
        
        # Load new AI player with different strategy
        ai_player = load_cfr_model(strategy_file)
        
        return jsonify({
            'success': True,
            'message': f'Strategy loaded from {strategy_file}',
            'model_loaded': ai_player.is_model_loaded()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to load strategy'
        }), 500


@app.route('/api/debug/info_state', methods=['GET'])
def debug_info_state():
    """Debug endpoint to see current information state"""
    if not game_engine or not game_engine.game_state:
        return jsonify({
            'success': False,
            'error': 'No active game'
        }), 400
    
    try:
        # Get info states for both players
        info_states = {}
        for player_id in [0, 1]:
            info_state = game_engine.get_information_set(player_id)
            info_states[f'player_{player_id}'] = info_state
        
        return jsonify({
            'success': True,
            'info_states': info_states,
            'current_player': game_engine.game_state.current_player,
            'street': game_engine.game_state.current_street.value
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("🂡" + "=" * 48 + "🂱")
    print("    🎰 SIMPLIFIED POKER vs CFR AI 🎰")
    print("🂡" + "=" * 48 + "🂱")
    print()
    print("🎮 Game Features:")
    print("   • 6 ranks (2,3,4,5,6,7) × 2 suits = 12 cards")
    print("   • 2 rounds only (Preflop → Flop)")
    print("   • Limit betting (2-4 structure)")
    print("   • Max 2 raises per round")
    print("   • 20 chip starting stacks")
    print()
    print("🧠 AI Features:")
    print("   • Loads trained MCCFR strategies")
    print("   • Direct info state lookup (no abstractions)")
    print("   • Fallback strategy if model not found")
    print("   • Automatic AI turn handling")
    print()
    print("📁 Expected strategy files:")
    print("   • ../mccfr/limit_holdem_strategy_parallel.pkl.gz")
    print("   • ../../mccfr/limit_holdem_strategy_parallel.pkl.gz")
    print()
    print("🌐 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    # Run on port 5001 instead of 5000 to avoid conflicts
    app.run(debug=True, host='0.0.0.0', port=5001)
