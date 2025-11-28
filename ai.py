import random

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
                end_pos[end_state] = curr_path
            
        moves = [
            (0, -1 , False), # left
            (0, 1, False), # right
            (0, 0, True), # rotate
            (1, 0, False) # down
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
                else:
                    new_path = curr_path + [2] # 2 for right

            new_state = (new_block.row_offset, new_block.col_offset, new_block.rotation_state)

            if game.block_fits(new_block) and new_state not in checked_pos:
                checked_pos.add(new_state)
                queue.append((new_block, new_path))
    
    return end_pos

# return a random move for the ai to take
def random_move(game):
    moves = list(get_all_end_positions(game).values())
    if moves:
        return random.choice(moves)
    return None # no valid moves found