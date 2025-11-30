import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import copy
import os
import sys
import pygame 
from collections import deque
from game import Game
from colors import Colors 
import ai

# --- Hyperparameters ---
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 20000 
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 1e-3
NUM_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 1000 
CHECKPOINT_FILE = "training_checkpoint.pth"

# --- VISUALIZATION SETTINGS ---
VISUAL_MODE = True      
RENDER_SPEED = 10       

# --- Curriculum Settings ---
SINGLE_PLAYER_EPISODES = 0

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, next_state, reward, done):
        self.memory.append((state, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_best_action(game, policy_net, epsilon):
    possible_moves = ai.get_all_end_positions(game)
    if not possible_moves:
        return None, None, None

    next_states = []
    move_info = [] 
    
    for block, path in possible_moves:
        temp_game = Game()
        temp_game.grid.grid = [row[:] for row in game.grid.grid] 
        
        tiles = block.get_cell_positions()
        for tile in tiles:
            temp_game.grid.grid[tile.row][tile.col] = block.id
            
        feats = ai.get_features(temp_game.grid, game.next_block)
        next_states.append(feats)
        move_info.append((block, path))

    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

    if random.random() < epsilon:
        idx = random.randint(0, len(next_states) - 1)
    else:
        with torch.no_grad():
            q_values = policy_net(next_states_tensor)
            idx = q_values.argmax().item()

    best_block, best_path = move_info[idx]
    best_feat = next_states_tensor[idx] 
    
    return best_block, best_path, best_feat

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    state_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)

    state_batch = torch.stack(state_batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in next_state_batch if s is not None])

    reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
    done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)

    current_q_values = policy_net(state_batch)
    
    next_q_values = torch.zeros(BATCH_SIZE, 1)
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            next_q_values[non_final_mask] = target_net(non_final_next_states)

    expected_q_values = reward_batch + (GAMMA * next_q_values)

    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def save_checkpoint(state, filename=CHECKPOINT_FILE):
    torch.save(state, filename)
    torch.save(state['policy_net_state_dict'], "trained_dqn.pth")
    print(f"\nCheckpoint saved to {filename} (and trained_dqn.pth)")

def draw_visuals(screen, game, episode, survived, reward, eps):
    screen.fill(Colors.dark_blue)
    game.draw(screen)
    
    font = pygame.font.Font(None, 30)
    texts = [
        f"Episode: {episode}",
        f"Survived: {survived}",
        f"Reward: {reward:.1f}",
        f"Epsilon: {eps:.2f}",
        f"Mode: {'Solo' if game.num_players == 1 else 'Versus'}"
    ]
    
    for i, text in enumerate(texts):
        surf = font.render(text, True, Colors.white)
        screen.blit(surf, (10, 10 + i * 25))
        
    pygame.display.update()

# --- Main Training Loop ---
policy_net = DQN(7, 1) 
target_net = DQN(7, 1)
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

start_episode = 0
steps_done = 0

if os.path.exists(CHECKPOINT_FILE):
    print(f"Loading checkpoint '{CHECKPOINT_FILE}'...")
    checkpoint = torch.load(CHECKPOINT_FILE)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    steps_done = checkpoint['steps_done']
    print(f"Resuming from Episode {start_episode}")
else:
    print("No checkpoint found. Starting fresh.")
    target_net.load_state_dict(policy_net.state_dict())

target_net.eval()

# --- Initialize Pygame for Visuals ---
if VISUAL_MODE:
    pygame.init()
    screen = pygame.display.set_mode((500, 620))
    pygame.display.set_caption("DQN Training Observer")
    clock = pygame.time.Clock()

print("Starting Curriculum Training (Ctrl+C to Save and Quit)...")

try:
    for episode in range(start_episode, NUM_EPISODES):
        game = Game()
        
        if episode < SINGLE_PLAYER_EPISODES:
            game.num_players = 1
            mode = "Solo"
        else:
            game.num_players = 2
            mode = "Versus"
        
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        
        total_reward = 0
        moves_survived = 0
        episode_steps = 0
        total_lines_cleared = 0
        
        last_state_features = None
        last_reward = 0
        
        while not game.game_over:
            if VISUAL_MODE:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt 
                
                draw_visuals(screen, game, episode, moves_survived, total_reward, epsilon)
                pygame.time.wait(RENDER_SPEED)

            steps_done += 1
            episode_steps += 1
            
            # --- PLAYER 1 (AI AGENT) ---
            if game.current_player_id == 0:
                block, path, chosen_state_features = get_best_action(game, policy_net, epsilon)
                
                if block:
                    game.current_block = block
                    game.lock_block()
                    moves_survived += 1
                    
                    if game.board_full:
                        game.game_over = True
                        reward = -20
                    else:
                        # Index 3 in features is "Complete Lines"
                        lines = int(chosen_state_features[3].item())
                        
                        total_lines_cleared += lines
                        
                        # Reward Shaping
                        survival_bonus = 1.0
                        height_penalty = chosen_state_features[4].item() * 0.05
                        reward = (lines ** 2) * 50 + survival_bonus - height_penalty
                        
                        if game.game_over:
                            reward = -100
                    
                    total_reward += reward
                    
                    if last_state_features is not None:
                        memory.push(last_state_features, chosen_state_features, last_reward, False)
                    
                    last_state_features = chosen_state_features
                    last_reward = reward

                    optimize_model(policy_net, target_net, memory, optimizer)
                else:
                    game.game_over = True
                    reward = -100
                    if last_state_features is not None:
                        memory.push(last_state_features, None, reward, True)

            # --- PLAYER 2 (SELF-PLAY OPPONENT) ---
            elif game.current_player_id == 1:
                 block, path, _ = get_best_action(game, policy_net, epsilon)
                 if block:
                     game.current_block = block
                     game.lock_block()
                     if game.board_full:
                         game.game_over = True
                 else:
                     game.game_over = True

            if episode_steps > MAX_STEPS_PER_EPISODE:
                game.game_over = True

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        print(f"Ep {episode} ({mode}): Survived {moves_survived} | Lines {total_lines_cleared} | Reward {total_reward:.1f} | Eps {epsilon:.2f}")

        if episode % 500 == 0:
            save_checkpoint({
                'episode': episode,
                'steps_done': steps_done,
                'policy_net_state_dict': policy_net.state_dict(),
                'target_net_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            })

    save_checkpoint({
        'episode': NUM_EPISODES,
        'steps_done': steps_done,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    })
    print("Training Complete.")

except KeyboardInterrupt:
    print("\nTraining interrupted by user!")
    save_checkpoint({
        'episode': episode, 
        'steps_done': steps_done,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    })
    if VISUAL_MODE:
        pygame.quit()
    print("Progress saved.")