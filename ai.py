import random
import torch
import torch.nn as nn

# dqn class
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

# load the model
trained_model = None

def load_model(filepath="trained_dqn.pth"):
    global trained_model
    try:
        model = DQN(7, 1) 
        model.load_state_dict(torch.load(filepath))
        model.eval() 
        trained_model = model
        print("AI Model loaded successfully!")
    except FileNotFoundError:
        print("No trained model found. AI will be random.")
        trained_model = None
    except RuntimeError:
        print("Model size mismatch, wrong number of features.")
        trained_model = None

# feature extraction
def get_features(grid, opponent_block=None):
    """
    Analyzes the grid and returns 7 features:
    [AggHeight, Holes, Bumpiness, Lines, MaxHeight, Wells, OppMaxLines]
    """
    
    heights = []
    for c in range(grid.num_cols):
        h = 0
        for r in range(grid.num_rows):
            if grid.grid[r][c] != 0:
                h = grid.num_rows - r
                break
        heights.append(h)

    # Aggregate Height
    aggregate_height = sum(heights)

    # Bumpiness
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])

    # Max Height
    max_height = max(heights) if heights else 0

    # Holes
    holes = 0
    for c in range(grid.num_cols):
        block_found = False
        for r in range(grid.num_rows):
            if grid.grid[r][c] != 0:
                block_found = True
            elif block_found and grid.grid[r][c] == 0:
                holes += 1
                
    # Cumulative Wells
    wells = 0
    for c in range(grid.num_cols):
        if c == 0: left_h = grid.num_rows 
        else: left_h = heights[c-1]
            
        if c == grid.num_cols - 1: right_h = grid.num_rows
        else: right_h = heights[c+1]
        
        if left_h > heights[c] and right_h > heights[c]:
            depth = min(left_h, right_h) - heights[c]
            wells += depth

    # Complete Lines (Self)
    complete_lines = 0
    for r in range(grid.num_rows):
        if all(grid.grid[r][c] != 0 for c in range(grid.num_cols)):
            complete_lines += 1

    # adversarial features
    opponent_max_lines = 0
    
    if opponent_block:
        # simulate dropping the opponent[s block in every possible column/rotation

        # check all 4 rotations
        for rot in range(4):
            temp_block = opponent_block.clone()
            for _ in range(rot):
                temp_block.rotate()
            
            # check all valid columns
            for col in range(-2, grid.num_cols):
                temp_block.col_offset = col
                temp_block.row_offset = 0
                
                # check if starting position is valid at the top
                if not _check_valid_position(grid, temp_block):
                    continue
                    
                # drop simulation
                landed = False
                for r in range(grid.num_rows):
                    temp_block.row_offset = r
                    if not _check_valid_position(grid, temp_block):
                        temp_block.row_offset = r - 1 # up one step
                        landed = True
                        break
                
                if not landed: 
                    # block landed on row 0
                    temp_block.row_offset = grid.num_rows - 1

                # if the final position is valid
                if _check_valid_position(grid, temp_block):
                    # calculate lines this move would clear
                    # check which rows the block occupies
                    tiles = temp_block.get_cell_positions()
                    lines_cleared_opp = 0
                    rows_affected = set(t.row for t in tiles)
                    
                    for r_check in rows_affected:
                        # check if row is full
                        row_full = True
                        for c_check in range(grid.num_cols):
                            cell_filled = (grid.grid[r_check][c_check] != 0)
                            for t in tiles:
                                if t.row == r_check and t.col == c_check:
                                    cell_filled = True
                            if not cell_filled:
                                row_full = False
                                break
                        if row_full:
                            lines_cleared_opp += 1
                    
                    if lines_cleared_opp > opponent_max_lines:
                        opponent_max_lines = lines_cleared_opp

    return [aggregate_height, holes, bumpiness, complete_lines, max_height, wells, opponent_max_lines]

# helper functions
def _check_valid_position(grid, block):
    tiles = block.get_cell_positions()
    for tile in tiles:
        if not grid.is_inside(tile.row, tile.col):
            return False
        if not grid.is_empty(tile.row, tile.col):
            return False
    return True

# BFS to find all the possible placements for the current block
def get_all_end_positions(game):
    end_pos = {}
    checked_pos = set()

    grid = game.grid
    
    start_block = game.current_block.clone() # clone so we're not repeatedly moving the same block
    queue = [(start_block, [])] 
    start_state = (game.current_block.row_offset, game.current_block.col_offset, game.current_block.rotation_state)
    checked_pos.add(start_state)
    while len(queue) > 0:
        curr_block, curr_path = queue.pop(0)
        
        # check if current state is an end placement, add to end positions if so
        test_down = curr_block.clone()
        test_down.move(1, 0)
        if not game.block_fits(test_down):
            end_state = (curr_block.row_offset, curr_block.col_offset, curr_block.rotation_state)
            if end_state not in end_pos:
                end_pos[end_state] = (curr_block, curr_path)
            
        moves = [
            (0, -1 , False), # left
            (0, 1, False), # right
            (0, 0, True), # rotate
            (1, 0, False), # down
            (0, 0, False) # do nothing
        ]

        for row_offset, col_offset, rotate in moves:
            new_block = curr_block.clone()

            new_block.move(1, 0) # have ai be affected by gravity

            if rotate:
                new_block.rotate()
                new_path = curr_path + [3] # 3 for rotate
            else:
                new_block.move(row_offset, col_offset)
                if row_offset:
                    new_path = curr_path + [0] # 0 for down
                elif col_offset == -1:
                    new_path = curr_path + [1] # 1 for left
                elif col_offset == 1:
                    new_path = curr_path + [2] # 2 for right
                else:
                    new_path = curr_path + [4] # 4 for do nothing

            new_state = (new_block.row_offset, new_block.col_offset, new_block.rotation_state)

            if game.block_fits(new_block) and new_state not in checked_pos:
                checked_pos.add(new_state)
                queue.append((new_block, new_path))
    
    return list(end_pos.values())

# return a random move for the ai to take
def random_move(game):
    moves = get_all_end_positions(game)
    if moves:
        return random.choice(moves)[1]
    return None # no valid moves found

def get_best_move(game):
    global trained_model
    if trained_model is None:
        return random_move(game)

    possible_moves = get_all_end_positions(game)
    if not possible_moves:
        return None

    best_score = -999999
    best_path = None
    
    for block, path in possible_moves:
        # create temp grid data
        temp_grid_data = [row[:] for row in game.grid.grid]
        tiles = block.get_cell_positions()
        
        # place block
        for tile in tiles:
             if 0 <= tile.row < len(temp_grid_data) and 0 <= tile.col < len(temp_grid_data[0]):
                temp_grid_data[tile.row][tile.col] = block.id
        
        # mock grid object for get_features
        class MockGrid:
            def __init__(self, data):
                self.grid = data
                self.num_rows = len(data)
                self.num_cols = len(data[0])
        
        feats = get_features(MockGrid(temp_grid_data))
        feats_t = torch.tensor(feats, dtype=torch.float32)
        
        with torch.no_grad():
            score = trained_model(feats_t).item()
            
        if score > best_score:
            best_score = score
            best_path = path

    return best_path