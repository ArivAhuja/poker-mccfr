
#!/usr/bin/env python3
"""
Simplified Poker GUI Startup Script - UPDATED VERSION
Loads your trained CFR models and starts the game interface
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'flask-cors', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_strategy_files():
    """Find your actual trained strategy files"""
    strategy_files = []
    
    # Updated paths to match your actual file structure
    search_paths = [
        "../mccfr/limit_holdem_strategy_parallel.pkl.gz",  # Your final model
        "../mccfr/checkpoints/mccfr_checkpoint.pkl.gz",   # Checkpoint
        "./gui_compatible_strategy.pkl.gz",               # Extracted version
        "./limit_holdem_strategy_parallel.pkl.gz",        # If moved here
    ]
    
    for file_path in search_paths:
        file_obj = Path(file_path)
        if file_obj.exists() and file_obj.is_file():
            strategy_files.append(file_obj.resolve())
            print(f"✅ Found strategy file: {file_path}")
    
    if not strategy_files:
        print("❌ No strategy files found")
        print("📁 Searched locations:")
        for path in search_paths:
            exists = "✅" if Path(path).exists() else "❌"
            print(f"   {exists} {path}")
        
        # Show what's actually in the directories
        mccfr_dir = Path("../mccfr")
        if mccfr_dir.exists():
            print(f"\n📂 Contents of {mccfr_dir}:")
            for item in mccfr_dir.iterdir():
                if item.is_file() and item.suffix in ['.gz', '.pkl']:
                    size_mb = item.stat().st_size / (1024*1024)
                    print(f"   📄 {item.name} ({size_mb:.1f} MB)")
                elif item.is_dir():
                    print(f"   📁 {item.name}/")
        
        checkpoints_dir = Path("../mccfr/checkpoints")
        if checkpoints_dir.exists():
            print(f"\n📂 Contents of {checkpoints_dir}:")
            for item in checkpoints_dir.iterdir():
                if item.is_file():
                    size_mb = item.stat().st_size / (1024*1024)
                    print(f"   📄 {item.name} ({size_mb:.1f} MB)")
    
    return strategy_files

def run_strategy_extractor():
    """Run the strategy extractor to create a GUI-compatible file"""
    print("\n🔧 Running strategy extractor...")
    print("   This will create a GUI-compatible version of your strategy file")
    
    try:
        # Create and run the extractor script
        extractor_code = '''
import sys
import os
import pickle
import gzip
from pathlib import Path

# Add current directory to path so we can import
sys.path.insert(0, ".")

# Your extraction logic here
def extract_and_save():
    """Extract strategies from your files"""
    
    # Find your actual strategy file
    strategy_file = None
    for path in ["../mccfr/limit_holdem_strategy_parallel.pkl.gz", 
                 "../mccfr/checkpoints/mccfr_checkpoint.pkl.gz"]:
        if os.path.exists(path):
            strategy_file = path
            break
    
    if not strategy_file:
        print("❌ No strategy file found for extraction")
        return False
    
    print(f"🔄 Extracting from: {strategy_file}")
    
    # Create dummy Config class to handle pickle loading
    class DummyConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __setstate__(self, state):
            self.__dict__.update(state)
    
    # Put it in main namespace for pickle
    import __main__
    __main__.Config = DummyConfig
    
    try:
        with gzip.open(strategy_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📋 Loaded data type: {type(data)}")
        
        # Extract strategy based on format
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            
            if 'final_strategy' in data:
                strategies = data['final_strategy']
                metadata = {'source': 'final_strategy', 'iterations': data.get('iterations')}
            elif 'strategy' in data:
                strategies = data['strategy'] 
                metadata = {'source': 'strategy', 'iteration': data.get('iteration')}
            elif 'avg_strategy_sum' in data:
                print("🔄 Converting avg_strategy_sum...")
                avg_strategy_sum = data['avg_strategy_sum']
                strategies = {}
                for info_state, strategy_sum in avg_strategy_sum.items():
                    total = sum(strategy_sum) if strategy_sum else 0
                    if total > 0:
                        strategies[info_state] = [s / total for s in strategy_sum]
                metadata = {'source': 'avg_strategy_sum_converted'}
            else:
                strategies = data
                metadata = {'source': 'direct'}
            
            # Save in simple format
            extracted_data = {'strategy': strategies, 'metadata': metadata}
            
            output_file = "gui_compatible_strategy.pkl.gz"
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(extracted_data, f)
            
            print(f"✅ Saved {len(strategies)} strategies to {output_file}")
            return True
            
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    extract_and_save()
'''
        
        # Write and run extractor
        with open("temp_extractor.py", "w") as f:
            f.write(extractor_code)
        
        import subprocess
        result = subprocess.run([sys.executable, "temp_extractor.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Clean up
        os.remove("temp_extractor.py")
        
        # Check if extraction succeeded
        if os.path.exists("gui_compatible_strategy.pkl.gz"):
            print("✅ Strategy extraction succeeded!")
            return True
        else:
            print("❌ Strategy extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Extractor failed: {e}")
        return False

def print_banner():
    """Print startup banner"""
    print("🂡" + "=" * 60 + "🂱")
    print("      🎰 SIMPLIFIED POKER vs TRAINED CFR AI 🎰")
    print("🂡" + "=" * 60 + "🂱")
    print()

def print_game_info():
    """Print game information"""
    print("🎮 Simplified Game Features:")
    print("   • 6 ranks (2,3,4,5,6,7) × 2 suits = 12 cards total")
    print("   • 2 rounds only: Preflop → Flop (no turn/river)")
    print("   • Fixed limit betting: $2-$4 structure")
    print("   • Max 2 raises per betting round")
    print("   • 20 chip starting stacks")
    print("   • Heads-up play (You vs AI)")
    print()
    
    print("🧠 CFR AI Features:")
    print("   • Loads your trained MCCFR strategies")
    print("   • Direct information state string lookup")
    print("   • No abstractions - exact game tree strategies")
    print("   • Fallback strategy if model not found")
    print()

def main():
    """Main startup function"""
    print_banner()
    print_game_info()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")
    
    # Find strategy files
    print("📁 Looking for trained strategy files...")
    strategy_files = find_strategy_files()
    
    if not strategy_files:
        print("\n🔧 No compatible strategy files found.")
        print("   Would you like to extract strategies from your pickle files?")
        
        extract = input("Extract strategies? (y/N): ").strip().lower()
        if extract == 'y':
            if run_strategy_extractor():
                # Re-scan for files
                strategy_files = find_strategy_files()
            else:
                print("❌ Extraction failed")
        
        if not strategy_files:
            print("⚠️  No strategy files available. AI will use fallback strategy.")
            use_fallback = input("Continue with fallback AI? (y/N): ").strip().lower()
            if use_fallback != 'y':
                print("👋 Exiting. Fix strategy file issues and try again.")
                sys.exit(0)
    
    if strategy_files:
        print("✅ Found strategy files:")
        for i, file in enumerate(strategy_files):
            file_size = file.stat().st_size / (1024*1024)  # MB
            print(f"   {i+1}. {file.name} ({file_size:.1f} MB)")
            print(f"      Path: {file}")
        print()
        
        # Ask user which file to use
        if len(strategy_files) > 1:
            try:
                choice = input(f"Choose strategy file (1-{len(strategy_files)}, Enter for first): ").strip()
                if choice:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(strategy_files):
                        selected_file = strategy_files[choice_idx]
                    else:
                        selected_file = strategy_files[0]
                else:
                    selected_file = strategy_files[0]
            except (ValueError, KeyboardInterrupt):
                selected_file = strategy_files[0]
        else:
            selected_file = strategy_files[0]
        
        print(f"🧠 Using strategy file: {selected_file.name}")
        
        # Set environment variable for the server to use
        os.environ['CFR_STRATEGY_FILE'] = str(selected_file)
    
    print()
    print("🚀 Starting simplified poker server...")
    print("🌐 Open your browser to: http://localhost:5001")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        # Import the server modules directly since we're in the gui folder
        from server import app
        import cfr_interface
        
        print("✅ Server loaded successfully!")
        print("🎮 Ready to play simplified poker vs CFR AI!")
        print()
        
        # Run the Flask development server on port 5001
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5001,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n")
        print("🛑 Server stopped by user")
        print("👋 Thanks for playing!")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure all files are in the correct directory:")
        print("   - backend/poker_engine.py")
        print("   - backend/cfr_interface.py") 
        print("   - backend/server.py")
        print("   - templates/index.html")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Check the error message above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
