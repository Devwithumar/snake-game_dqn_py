"""
AI-powered Snake game with a DQN (Deep Q-Network) agent and pygame visuals.

Features:
- Modern-looking pygame GUI with grid and smooth animations
- DQN agent implemented with PyTorch (MLP) using experience replay
- State encoding with danger flags and food direction
- Training loop with epsilon-greedy and target network updates
- Save/load model and replay memory
- Three modes:
  - `train` (train agent)
  - `play` (watch trained agent)
  - `human` (play yourself with arrow keys)

Dependencies:
- Python 3.8+
- pygame
- numpy
- torch

Run examples:
- Train: python snake_dqn.py --mode train
- Play AI: python snake_dqn.py --mode play --model ./models/dqn_snake.pth
- Play yourself: python snake_dqn.py --mode human
"""

import argparse
import math
import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------
# --- Hyperparameters -----
# --------------------------
GRID_SIZE = 20  # size of cells in pixels
GRID_W = 30     # number of columns
GRID_H = 20     # number of rows
WINDOW_W = GRID_SIZE * GRID_W
WINDOW_H = GRID_SIZE * GRID_H
FPS = 30

# DQN hyperparams
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 50_000
MIN_REPLAY_SIZE = 2000
TARGET_UPDATE_FREQ = 1000  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100_000  # steps

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# --- Utilities ----------
# --------------------------
Point = namedtuple('Point', 'x y')

def point_add(a, b):
    return Point(a.x + b[0], a.y + b[1])

DIRECTIONS = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

# --------------------------
# --- Game environment ----
# --------------------------
class SnakeGameAI:
    def __init__(self, w=GRID_W, h=GRID_H):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.head = Point(self.w // 2, self.h // 2)
        self.direction = 0  # start moving right
        self.snake = deque([self.head,
                            Point(self.head.x - 1, self.head.y),
                            Point(self.head.x - 2, self.head.y)])
        self._place_food()
        self.frame_iteration = 0
        self.score = 0
        return self.get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            food = Point(x, y)
            if food not in self.snake:
                self.food = food
                return

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # walls
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        # self
        if pt in list(self.snake)[1:]:
            return True
        return False

    def step(self, action):
        # action: [straight, right turn, left turn] relative to current direction
        self.frame_iteration += 1

        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        move = DIRECTIONS[self.direction]
        self.head = point_add(self.head, move)

        reward = 0
        game_over = False

        if self.is_collision(self.head) or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over, self.score

        self.snake.appendleft(self.head)

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # small living penalty to encourage shorter paths
        reward += -0.01

        return self.get_state(), reward, game_over, self.score

    def get_state(self):
        head = self.head

        # points for straight, right, left based on current direction
        dir_vector = DIRECTIONS[self.direction]
        left_dir = DIRECTIONS[(self.direction - 1) % 4]
        right_dir = DIRECTIONS[(self.direction + 1) % 4]

        p_straight = point_add(head, dir_vector)
        p_right = point_add(head, right_dir)
        p_left = point_add(head, left_dir)

        danger_straight = int(self.is_collision(p_straight))
        danger_right = int(self.is_collision(p_right))
        danger_left = int(self.is_collision(p_left))

        dir_up = int(self.direction == 3)
        dir_down = int(self.direction == 1)
        dir_left = int(self.direction == 2)
        dir_right = int(self.direction == 0)

        food_left = int(self.food.x < head.x)
        food_right = int(self.food.x > head.x)
        food_up = int(self.food.y < head.y)
        food_down = int(self.food.y > head.y)

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down,
        ], dtype=np.float32)

        return state

# --------------------------
# --- Neural Network -----
# --------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# --- Replay Memory ------
# --------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# --------------------------
# --- Agent --------------
# --------------------------
class Agent:
    def __init__(self, state_dim=11, n_actions=3):
        self.n_actions = n_actions
        self.policy_net = DQN(state_dim, n_actions).to(DEVICE)
        self.target_net = DQN(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, train=True):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if train and random.random() < eps_threshold:
            a = random.randrange(self.n_actions)
            action = np.zeros(self.n_actions, dtype=np.float32)
            action[a] = 1.0
            return action, a
        else:
            with torch.no_grad():
                state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_v)
                a = q_values.max(1)[1].item()
                action = np.zeros(self.n_actions, dtype=np.float32)
                action[a] = 1.0
                return action, a

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*transitions)

        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(DEVICE)
        action_batch = torch.tensor([np.argmax(a) for a in batch.action], dtype=torch.int64).to(DEVICE)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE)
        next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32).to(DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE)

        q_values = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]
            expected_state_action_values = reward_batch + (1.0 - done_batch) * GAMMA * max_next_q_values

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(data['policy_state_dict'])
        self.target_net.load_state_dict(data.get('target_state_dict', data['policy_state_dict']))
        self.optimizer.load_state_dict(data.get('optimizer_state_dict', self.optimizer.state_dict()))
        self.steps_done = data.get('steps_done', 0)

# --------------------------
# --- Pygame Renderer -----
# --------------------------
class Renderer:
    def __init__(self, game: SnakeGameAI):
        pygame.init()
        pygame.display.set_caption('Snake DQN')
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 18)
        self.large_font = pygame.font.SysFont('Consolas', 28)
        self.game = game

    def draw(self, score, episode, eps, loss, mode='train'):
        self.screen.fill((10, 10, 10))

        for x in range(0, WINDOW_W, GRID_SIZE):
            pygame.draw.line(self.screen, (20, 20, 20), (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, GRID_SIZE):
            pygame.draw.line(self.screen, (20, 20, 20), (0, y), (WINDOW_W, y))

        f = self.game.food
        pygame.draw.rect(self.screen, (200, 50, 50), (f.x * GRID_SIZE + 2, f.y * GRID_SIZE + 2, GRID_SIZE - 4, GRID_SIZE - 4), border_radius=6)

        for i, p in enumerate(self.game.snake):
            t = i / max(1, len(self.game.snake) - 1)
            r = int(30 + t * 180)
            g = int(180 - t * 100)
            b = int(30 + (1 - t) * 120)
            rect = pygame.Rect(p.x * GRID_SIZE + 2, p.y * GRID_SIZE + 2, GRID_SIZE - 4, GRID_SIZE - 4)
            pygame.draw.rect(self.screen, (r, g, b), rect, border_radius=6)

        hud = f"Score: {score}  Episode: {episode}  Eps: {eps:.3f}  Mode: {mode}"
        text = self.font.render(hud, True, (220, 220, 220))
        self.screen.blit(text, (10, 10))

        if loss is not None:
            loss_text = self.font.render(f"Loss: {loss:.4f}", True, (220, 220, 220))
            self.screen.blit(loss_text, (10, 35))

        pygame.display.flip()
        self.clock.tick(FPS)

# --------------------------
# --- Training Loop ------
# --------------------------
def train(agent: Agent, n_episodes=10000, render_every=50):
    game = SnakeGameAI()
    renderer = Renderer(game)

    total_steps = 0
    losses = []
    episode_rewards = deque(maxlen=100)

    print("Filling replay memory...")
    state = game.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action_idx = random.randrange(agent.n_actions)
        action = np.zeros(agent.n_actions, dtype=np.float32)
        action[action_idx] = 1.0
        next_state, reward, done, _ = game.step(action)
        agent.memory.push(state, action, reward, next_state, float(done))
        state = next_state if not done else game.reset()

    print("Starting training...")
    for ep in range(1, n_episodes + 1):
        state = game.reset()
        ep_reward = 0
        loss_val = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action, action_idx = agent.select_action(state, train=True)
            next_state, reward, done, score = game.step(action)
            ep_reward += reward

            agent.memory.push(state, action, reward, next_state, float(done))
            state = next_state

            loss = agent.optimize_model()
            if loss is not None:
                loss_val = loss
                losses.append(loss)

            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            if done:
                episode_rewards.append(score)
                break

        if ep % render_every == 0:
            avg_score = np.mean(episode_rewards) if len(episode_rewards) else 0.0
            print(f"Episode {ep} | Score: {score} | AvgScore(100): {avg_score:.2f} | Eps: {max(EPS_END, EPS_START * math.exp(-agent.steps_done / EPS_DECAY)):.3f} | Loss: {loss_val}")
            renderer.draw(score, ep, max(EPS_END, EPS_START * math.exp(-agent.steps_done / EPS_DECAY)), loss_val, mode='train')
            agent.save(os.path.join(MODEL_DIR, 'dqn_snake.pth'))

    agent.save(os.path.join(MODEL_DIR, 'dqn_snake_final.pth'))
    pygame.quit()

# --------------------------
# --- Play (inference) ----
# --------------------------
def play(agent: Agent, model_path=None):
    if model_path:
        agent.load(model_path)
    game = SnakeGameAI()
    renderer = Renderer(game)
 

    try:
        while True:
            state = game.reset()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                action, idx = agent.select_action(state, train=False)
                next_state, reward, done, score = game.step(action)
                state = next_state
                renderer.draw(score, episode=0, eps=0.0, loss=None, mode='play')
                if done:
                    time.sleep(0.5)
                    break
    except KeyboardInterrupt:
        pygame.quit()

# --------------------------
