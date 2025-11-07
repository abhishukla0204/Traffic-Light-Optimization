# Traffic Light Optimization - System Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation Guide](#installation-guide)
3. [Configuration](#configuration)
4. [Running the Project](#running-the-project)
5. [System Architecture](#system-architecture)
6. [Technical Specifications](#technical-specifications)
7. [Troubleshooting](#troubleshooting)
8. [Development Guide](#development-guide)

---

## System Overview

### Purpose

This system uses Deep Reinforcement Learning to optimize traffic light control at a 4-way intersection. The AI agent learns to minimize vehicle waiting times by observing traffic patterns and making intelligent decisions about when to change traffic signals.

### Technology Stack

- **Language**: Python 3.12
- **Deep Learning**: TensorFlow 2.20 / Keras 3.12
- **Simulation**: SUMO 1.24 (Simulation of Urban MObility)
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn

### System Requirements

**Minimum:**
- CPU: Intel Core i5 or equivalent
- RAM: 8GB
- Storage: 2GB free space
- OS: Windows 10/11, Linux, macOS

**Recommended:**
- CPU: Intel Core i7 or equivalent
- RAM: 16GB
- Storage: 5GB free space
- GPU: NVIDIA GPU with CUDA support (optional, for faster training)

---

## Installation Guide

### Step 1: Install Python

Download and install Python 3.12 or higher from [python.org](https://www.python.org/downloads/)

**Verify installation:**
```bash
python --version
# Should output: Python 3.12.x
```

### Step 2: Install SUMO

#### Windows:
1. Download SUMO from [eclipse.org/sumo](https://www.eclipse.org/sumo/)
2. Run the installer (default location: `C:\Program Files (x86)\Eclipse\Sumo`)
3. Set environment variables:
   ```powershell
   setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
   # Restart terminal
   ```

#### Linux (Ubuntu/Debian):
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
```

#### macOS:
```bash
brew install sumo
export SUMO_HOME="/opt/homebrew/share/sumo"
echo 'export SUMO_HOME="/opt/homebrew/share/sumo"' >> ~/.zshrc
```

**Verify SUMO installation:**
```bash
sumo --version
# Should output: Eclipse SUMO sumo Version 1.24.0
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/sumitsingh3072/Traffic-Light-Optimization.git
cd Traffic-Light-Optimization
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r Requirements.txt
```

### Step 6: Verify Installation

Create a file `verify.py`:
```python
import tensorflow as tf
import traci
import sumolib
import numpy as np

print("✓ TensorFlow:", tf.__version__)
print("✓ NumPy:", np.__version__)
print("✓ SUMO tools imported successfully")
print("\nSetup complete!")
```

Run:
```bash
python verify.py
```

---

## Configuration

### Training Configuration (`TCLS/training_settings.ini`)

```ini
[simulation]
gui = False                  # Show SUMO GUI during training
total_episodes = 100         # Number of training episodes
max_steps = 5400            # Steps per episode (90 min simulated)
n_cars_generated = 1000     # Number of cars per episode
green_duration = 10         # Green light duration (seconds)
yellow_duration = 4         # Yellow light duration (seconds)

[model]
num_layers = 4              # Number of hidden layers (+ 1 input = 5)
width_layers = 400          # Neurons per hidden layer
batch_size = 100            # Training batch size
learning_rate = 0.001       # Adam optimizer learning rate
training_epochs = 800       # Training iterations per episode

[memory]
memory_size_min = 600       # Minimum samples before training
memory_size_max = 50000     # Maximum experience buffer size

[agent]
num_states = 80             # State space dimension
num_actions = 4             # Action space dimension
gamma = 0.75                # Discount factor for future rewards

[dir]
models_path_name = models   # Directory to save trained models
sumocfg_file_name = sumo_config.sumocfg  # SUMO config file
```

### Testing Configuration (`TCLS/testing_settings.ini`)

```ini
[simulation]
gui = True                  # Show SUMO GUI during testing
max_steps = 5400           # Steps for test episode
n_cars_generated = 1000    # Number of cars
episode_seed = 10000       # Random seed (for reproducibility)
yellow_duration = 4        # Yellow light duration
green_duration = 10        # Green light duration

[agent]
num_states = 80            # State space dimension
num_actions = 4            # Action space dimension

[dir]
models_path_name = models  # Directory containing trained models
sumocfg_file_name = sumo_config.sumocfg
model_to_test = 5         # Which model to test (model_X)
```

### Customization Tips

**For Faster Training:**
- Set `total_episodes = 25` (quick test)
- Set `n_cars_generated = 500` (less traffic)
- Set `training_epochs = 400` (fewer updates)

**For Better Performance:**
- Set `total_episodes = 200` (more learning)
- Set `width_layers = 800` (larger network)
- Set `learning_rate = 0.0001` (finer adjustments)

**For Debugging:**
- Set `gui = True` (watch simulation)
- Set `total_episodes = 1` (single episode)
- Set `max_steps = 1000` (shorter episodes)

---

## Running the Project

### Training a New Model

#### Basic Training:
```bash
cd TCLS
python training_main.py
```

This will:
1. Generate random traffic patterns
2. Run 100 episodes of simulation
3. Train the neural network after each episode
4. Save the model in `TCLS/models/model_X/`
5. Take approximately 6-8 hours

#### Training with Visualization:
Edit `training_settings.ini`:
```ini
[simulation]
gui = True
```

Then run training. You'll see the SUMO window with live traffic.

#### Quick Training Test:
Edit `training_settings.ini`:
```ini
[simulation]
total_episodes = 10
```

Trains faster (~30-40 minutes) for testing purposes.

### Testing a Trained Model

#### Step 1: Identify Model Number
```bash
ls TCLS/models/
# Output: model_1  model_2  model_3  ...
```

#### Step 2: Configure Test
Edit `testing_settings.ini`:
```ini
[dir]
model_to_test = 5  # Replace with your model number
```

#### Step 3: Run Test
```bash
cd TCLS
python testing_main.py
```

This will:
1. Load the trained model
2. Run 1 test episode with GUI
3. Generate performance graphs
4. Save results in `models/model_X/test/`

### Viewing Results

After training or testing, check:

**Training Results** (`TCLS/models/model_X/`):
- `trained_model.h5` - Trained neural network
- `training_settings.ini` - Configuration used
- Performance plots (if generated)

**Testing Results** (`TCLS/models/model_X/test/`):
- `plot_reward.png` - Reward per action
- `plot_queue.png` - Queue length over time
- `plot_reward_data.txt` - Raw reward data
- `plot_queue_data.txt` - Raw queue data
- `testing_settings.ini` - Test configuration

### Command Reference

```bash
# Setup
python -m venv .venv              # Create virtual environment
.venv\Scripts\activate            # Activate (Windows)
source .venv/bin/activate         # Activate (Linux/Mac)
pip install -r Requirements.txt   # Install dependencies

# Training
cd TCLS
python training_main.py           # Start training

# Testing
python testing_main.py            # Test trained model

# Utilities
python -c "import tensorflow; print(tensorflow.__version__)"  # Check TF
sumo --version                    # Check SUMO
```

---

## System Architecture

### Component Overview

```
┌────────────────────────────────────────────┐
│               Training System              │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────┐      ┌──────────────┐    │
│  │   SUMO       │◄────►│  TraCI API   │    │
│  │  Simulator   │      │  (Control)   │    │
│  └──────────────┘      └──────┬───────┘    │
│         ▲                     │            │
│         │             ┌───────▼────────┐   │
│         │             │   Simulation   │   │
│         │             │     Logic      │   │
│         │             └───────┬────────┘   │
│         │                     │            │
│    ┌────▼─────────────────────▼───────┐    │
│    │      Q-Learning Agent            │    │
│    │  ┌────────────────────────────┐  │    │
│    │  │    Neural Network          │  │    │
│    │  │  80 → 400 → ... → 400 → 4  │  │    │
│    │  └────────────────────────────┘  │    │
│    │  ┌────────────────────────────┐  │    │
│    │  │   Experience Replay        │  │    │
│    │  │   Memory (50k samples)     │  │    │
│    │  └────────────────────────────┘  │    │
│    └──────────────────────────────────┘    │
│                                            │
└────────────────────────────────────────────┘
```

### Data Flow

1. **Traffic Generation** (`generator.py`):
   - Creates random vehicle routes using Weibull distribution
   - Generates `episode_routes.rou.xml` file
   - 75% straight traffic, 25% turning traffic

2. **Simulation** (`training_simulation.py`):
   - SUMO simulates vehicle movement
   - Agent observes state every 10-14 seconds
   - Agent selects traffic light phase
   - Reward calculated based on waiting time change

3. **Learning** (`model.py` + `memory.py`):
   - Experience stored: (state, action, reward, next_state)
   - Neural network predicts Q-values
   - Q-values updated using Bellman equation
   - Network trained on random batches from memory

4. **Visualization** (`visualization.py`):
   - Plots reward progression
   - Plots queue length over time
   - Saves data to text files

### State Representation

The intersection is discretized into an 80-dimensional state vector:

```
State Space Encoding:
┌─────────────────────────────────────────┐
│  Direction  │ Lane Type │ Distance Cells│
├─────────────┼───────────┼───────────────┤
│   North     │  Regular  │   10 cells    │  → Positions 0-9
│   North     │  Left     │   10 cells    │  → Positions 10-19
│   East      │  Regular  │   10 cells    │  → Positions 20-29
│   East      │  Left     │   10 cells    │  → Positions 30-39
│   South     │  Regular  │   10 cells    │  → Positions 40-49
│   South     │  Left     │   10 cells    │  → Positions 50-59
│   West      │  Regular  │   10 cells    │  → Positions 60-69
│   West      │  Left     │   10 cells    │  → Positions 70-79
└─────────────┴───────────┴───────────────┘

Distance Cells (from intersection):
Cell 0: 0-7m     Cell 5: 40-60m
Cell 1: 7-14m    Cell 6: 60-100m
Cell 2: 14-21m   Cell 7: 100-160m
Cell 3: 21-28m   Cell 8: 160-400m
Cell 4: 28-40m   Cell 9: 400-750m
```

### Action Mapping

```
Action 0 (NS Green):     Action 1 (NSL Green):
    ↑ →                      ↑ ←
  ← █ →                    → █ ←
    ↓ ←                      ↓ →

Action 2 (EW Green):     Action 3 (EWL Green):
    ↑ ←                      ↑ →
  → █ ←                    ← █ →
    ↓ →                      ↓ ←

Legend: █ = Intersection, Green arrows = allowed movements
```

---

## Technical Specifications

### Deep Q-Learning Algorithm

**Q-Value Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

Where:
- Q(s,a) = Quality of action a in state s
- α = Learning rate (handled by Adam optimizer)
- r = Immediate reward
- γ = 0.75 (discount factor)
- s' = Next state
- a' = Next action
```

**Epsilon-Greedy Policy:**
```
ε(episode) = 1.0 - (episode / total_episodes)

Action selection:
- With probability ε: Random action (exploration)
- With probability 1-ε: argmax Q(s,a) (exploitation)
```

**Experience Replay:**
```
Memory stores tuples: (s, a, r, s')
Training samples random batches to:
- Break temporal correlations
- Reuse past experiences
- Stabilize learning
```

### Neural Network Details

**Architecture:**
```python
model = Sequential([
    Dense(400, activation='relu', input_dim=80),  # Layer 1
    Dense(400, activation='relu'),                 # Layer 2
    Dense(400, activation='relu'),                 # Layer 3
    Dense(400, activation='relu'),                 # Layer 4
    Dense(400, activation='relu'),                 # Layer 5
    Dense(4, activation='linear')                  # Output
])
```

**Training:**
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (learning_rate=0.001)
- Batch Size: 100 samples
- Epochs: 800 per episode

**Total Parameters:** ~802,804 trainable parameters

### Performance Characteristics

**Training:**
- Episode Duration: ~3-4 minutes (simulation + training)
- Simulation Time: ~10-40 seconds
- Training Time: ~200 seconds
- Total Training (100 episodes): ~6-8 hours

**Inference:**
- Prediction Time: 30-50ms per decision
- Real-time Capable: Yes (decisions every 10-14 seconds)
- Memory Usage: ~100MB (model loaded)

---

## Troubleshooting

### Common Issues

#### 1. "SUMO_HOME not found"

**Problem:** SUMO environment variable not set

**Solution:**
```bash
# Windows
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
# Restart terminal

# Linux/Mac
export SUMO_HOME="/usr/share/sumo"  # Or your SUMO path
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

#### 2. "ModuleNotFoundError: No module named 'traci'"

**Problem:** Python can't find SUMO tools

**Solution:**
```bash
# Windows PowerShell
$env:PYTHONPATH = "C:\Program Files (x86)\Eclipse\Sumo\tools"

# Linux/Mac
export PYTHONPATH="$SUMO_HOME/tools"
```

#### 3. "Model number not found"

**Problem:** Testing wrong model number

**Solution:**
```bash
# Check available models
ls TCLS/models/

# Edit testing_settings.ini
[dir]
model_to_test = X  # Use correct model number
```

#### 4. Training crashes or freezes

**Possible causes:**
- Out of memory
- Disk space full
- SUMO process stuck

**Solutions:**
- Reduce `n_cars_generated` in settings
- Reduce `memory_size_max`
- Check disk space: `df -h` (Linux) or `Get-PSDrive` (Windows)
- Kill stuck SUMO processes: `killall sumo` (Linux) or Task Manager (Windows)

#### 5. "ImportError: You must install graphviz"

**Problem:** Graphviz system tool not installed (for model visualization)

**Solution:**
- This is optional (model still works)
- Install: https://graphviz.org/download/
- Or ignore (only affects architecture diagram)

### Debug Mode

To debug issues, modify settings for verbose output:

**Enable SUMO GUI:**
```ini
[simulation]
gui = True
```

**Reduce Episode Length:**
```ini
[simulation]
total_episodes = 1
max_steps = 1000
```

**Python Debug Mode:**
```python
# Add to training_main.py at top
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Development Guide

### Project Structure Explained

```
TCLS/
├── training_main.py          # Entry point for training
│   └── Orchestrates: Generator → Simulation → Training → Saving
│
├── testing_main.py           # Entry point for testing
│   └── Loads model → Runs simulation → Saves results
│
├── model.py                  # Neural network classes
│   ├── TrainModel           # For training (includes training methods)
│   └── TestModel            # For testing (inference only)
│
├── training_simulation.py    # Training simulation loop
│   ├── run()                # Main simulation loop
│   ├── _simulate()          # Step through SUMO
│   ├── _get_state()         # Extract state from SUMO
│   ├── _choose_action()     # Epsilon-greedy policy
│   ├── _set_green_phase()   # Apply action to SUMO
│   └── _replay()            # Train neural network
│
├── testing_simulation.py     # Testing simulation loop
│   └── Similar to training but no learning
│
├── generator.py              # Traffic generation
│   └── generate_routefile() # Creates episode_routes.rou.xml
│
├── memory.py                 # Experience replay buffer
│   ├── add_sample()         # Store experience
│   └── get_samples()        # Retrieve random batch
│
├── utils.py                  # Helper functions
│   ├── import_train_configuration()
│   ├── import_test_configuration()
│   ├── set_sumo()           # Configure SUMO command
│   ├── set_train_path()     # Create model directory
│   └── set_test_path()      # Create test directory
│
└── visualization.py          # Plotting utilities
    └── save_data_and_plot() # Create graphs and save data
```

### Adding New Features

#### Add New Action

1. **Define in `training_simulation.py`:**
```python
PHASE_NEW_ACTION = 8  # New phase code
```

2. **Update `_set_green_phase()`:**
```python
elif action_number == 4:
    traci.trafficlight.setPhase("TL", PHASE_NEW_ACTION)
```

3. **Update configuration:**
```ini
[agent]
num_actions = 5  # Increase from 4
```

4. **Update SUMO network** (`intersection/environment.net.xml`)

#### Modify Reward Function

Edit `training_simulation.py`, in the `run()` method:

```python
# Current reward
reward = old_total_wait - current_total_wait

# Example: Add queue length penalty
queue_penalty = self._get_queue_length() * 0.1
reward = old_total_wait - current_total_wait - queue_penalty
```

#### Change Network Architecture

Edit `model.py`, in `TrainModel._build_model()`:

```python
# Current: 5 layers x 400 neurons
# Change to 3 layers x 200 neurons:
def _build_model(self, num_layers, width):
    inputs = keras.Input(shape=(self._input_dim,))
    x = layers.Dense(200, activation='relu')(inputs)
    for _ in range(2):  # 2 more hidden layers
        x = layers.Dense(200, activation='relu')(x)
    outputs = layers.Dense(self._output_dim, activation='linear')(x)
    # ...
```

### Testing Modifications

```bash
# Quick test with minimal settings
# Edit training_settings.ini:
total_episodes = 2
max_steps = 1000
n_cars_generated = 100
gui = True

# Run and observe
python training_main.py
```

### Best Practices

1. **Always use virtual environment**
2. **Backup trained models** before experimenting
3. **Document changes** in code comments
4. **Test with small episodes** before full training
5. **Track hyperparameters** in settings files
6. **Version control** with meaningful commit messages

---

## Additional Resources

### SUMO Documentation
- Official Docs: https://sumo.dlr.de/docs/
- TraCI Interface: https://sumo.dlr.de/docs/TraCI.html
- Network Files: https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html

### Deep Q-Learning
- Original Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Tutorial: https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

### TensorFlow/Keras
- TensorFlow Docs: https://www.tensorflow.org/api_docs
- Keras Guide: https://keras.io/guides/

---

## Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Open an issue on GitHub
4. Contact: sumitsingh3072@github

---

**Document Version:** 1.0  
**Last Updated:** November 7, 2025  
**Author:** System Documentation Team
