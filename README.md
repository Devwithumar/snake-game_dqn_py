# snake-game_dqn_py
# 🐍 Snake Game with Deep Q-Learning AI

An AI-powered Snake game built with **PyGame** for visuals and **PyTorch** for the reinforcement learning agent.  
Play it yourself, watch the AI learn, or see it perform after training.

---

## 🎮 Features
- **Three Modes**  
  - 🧑 Human: Play the classic Snake game yourself with arrow keys.  
  - 🤖 Train: Let a DQN agent learn how to play using reinforcement learning.  
  - 🎥 Play: Watch a trained model play Snake on its own.  

- **Deep Q-Learning (DQN)**  
  - Replay buffer  
  - Epsilon-greedy exploration  
  - Neural network function approximation  

- **Clean Visuals**  
  - Built with PyGame  
  - Grid-based snake graphics  
  - Score counter  

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/snake-dqn-game.git

##Recommended: use Python 3.10 or 3.11 (PyTorch isn’t fully stable on Python 3.13 yet).
🚀 Usage
1. Train the AI
'python snake_dqn.py --mode train --episodes 2000' (in terminal)

2. Play Yourself
'python snake_dqn.py --mode human' (in terminal)
Control the snake using your arrow keys.

3. Watch the AI Play
'python snake_dqn.py --mode play --model ./models/dqn_snake.pth' (in terminal)
cd snake-dqn-game
pip install -r requirements.txt
