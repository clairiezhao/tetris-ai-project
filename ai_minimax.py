# ai_minimax.py

import random
import copy
from math import inf

# We reuse the BFS-based move generator from the existing ai.py module.
# get_all_end_positions(game) returns a dict mapping final block states
# to a *path* (list of ints) describing how to move/rotate the current
# block to reach that state.
from ai import get_all_end_positions


def board_heuristic(grid, lines_cleared):
    """
    Heuristic value for the shared board.
    Negative is bad (tall, holey, bumpy), positive is good.
    """
    num_rows = grid.num_rows
    num_cols = grid.num_cols
    cells = grid.grid  # 2D list

    heights = [0] * num_cols
    holes = 0

    # Compute column heights and holes
    for col in range(num_cols):
        seen_block = False
        col_height = 0
        col_holes = 0
        for row in range(num_rows):
            if cells[row][col] != 0:
                if not seen_block:
                    seen_block = True
                    col_height = num_rows - row
            elif seen_block:
                # empty cell below a filled cell -> hole
                col_holes += 1
        heights[col] = col_height
        holes += col_holes

    max_height = max(heights)
    agg_height = sum(heights)
    bumpiness = sum(abs(heights[c] - heights[c + 1]) for c in range(num_cols - 1))

    # Tune these weights to taste
    return (
        50 * lines_cleared
        - 0.5 * agg_height
        - 2 * holes
        - 0.3 * bumpiness
        - 1.5 * max_height
    )


def utility(game, root_player_id, lines_cleared):
    """
    U = weight * (myScore - oppScore) + board heuristic.
    """
    p1_score = game.player1.score
    p2_score = game.player2.score
    if root_player_id == 0:
        score_diff = p1_score - p2_score
        player = game.player1
        opponent = game.player2
    else:
        score_diff = p2_score - p1_score
        player = game.player2
        opponent = game.player1

    board_val = board_heuristic(game.grid, lines_cleared)

    # Dynamically prioritize lines cleared depending on place in the game
    lines_val = 50 * lines_cleared
    if player.score + p2_score < 500:
        lines_val *= 2

    opp_val = 0
    if game.next_block is not None:
        moves_dict = get_all_end_positions(game)
        if moves_dict:
            max_opp_score = -inf
            for item in moves_dict:
                path = item[1]
                g_copy = copy_game(game)
                g_after, opp_lines_cleared = simulate_path(g_copy, path)
                score = board_heuristic(g_after.grid, opp_lines_cleared)
                max_opp_score = max(max_opp_score, score)
            opp_val = 0.5 * max_opp_score

    # Make score difference matter a lot, but still care about board shape
    return 0.5 * score_diff + board_val + lines_val - opp_val



def simulate_path(game, path):
    """
    Given a *copy* of Game and a move path (as produced by get_all_end_positions),
    apply all moves for the current player and let the block lock in place.

    The encoding of moves is the same as in gameDisplay.py:
        0 -> move_down()
        1 -> move_left()
        2 -> move_right()
        anything else -> rotate()
    """
    # Remember which player is currently moving so we can detect when the
    # block has locked and the turn has switched.
    start_player_id = game.current_player_id

    # Track previous score
    initial_score = game.player1.score if start_player_id == 0 else game.player2.score

    # Apply the scripted moves.
    for move in path:
        if game.game_over:
            break
        if move == 0:
            game.move_down()
        elif move == 1:
            game.move_left()
        elif move == 2:
            game.move_right()
        else:
            game.rotate()

    # Ensure the block actually locks (if it hasn't already) by repeatedly
    # moving it down until the turn changes or the game ends.
    # This mirrors what the main game loop does over multiple ticks.
    while (not game.game_over) and (game.current_player_id == start_player_id):
        game.move_down()

    # Calculate lines cleared using previous score and final score
    final_score = game.player1.score if start_player_id == 0 else game.player2.score
    lines_cleared = 0
    if final_score - initial_score == 40:
        lines_cleared = 1
    elif final_score - initial_score == 100:
        lines_cleared = 2
    elif final_score - initial_score == 300:
        lines_cleared = 3
    elif final_score - initial_score == 1200:
        lines_cleared = 4

    return game, lines_cleared


def copy_game(game):
    """
    Create a deep copy of the Game state for search.

    This relies on the Game / Grid / Block / Player objects being
    reasonably deepcopy-able. If you run into issues, you may want
    to implement an explicit Game.clone() method and call that here.
    """
    return game.clone()


def minimax_value(game, depth, root_player_id, maximizing, branch_limit=8, alpha=-inf, beta=inf):
    """
    Recursive minimax with alpha-beta pruning

    maximizing is True if it's player's move (maximize score), False if it's opponent's move (minimize score)
    """
    if depth <= 0 or game.game_over:
        return utility(game, root_player_id, 0)

    moves_dict = get_all_end_positions(game)
    if not moves_dict:
        return utility(game, root_player_id, 0)

    # Rank child moves by immediate heuristic and keep only top K
    scored_children = []
    for item in moves_dict:
        path = item[1]
        g_copy = copy_game(game)
        g_after, lines_cleared = simulate_path(g_copy, path)
        score = utility(g_after, root_player_id, lines_cleared)
        scored_children.append((score, path, g_after, lines_cleared))

    # Sort descending for MAX, ascending for MIN, then cut to branch_limit
    scored_children.sort(key=lambda x: x[0], reverse=maximizing)
    scored_children = scored_children[:branch_limit]

    # Max player
    if maximizing:
        value = -inf
        for _, _, g_after, lines_cleared in scored_children:
            if depth == 1 or g_after.game_over:
                child_val = utility(g_after, root_player_id, lines_cleared)
            else:
                child_val = minimax_value(g_after, depth - 1, root_player_id, False, branch_limit, alpha, beta)
            
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else: # Min player
        value = inf
        for _, _, g_after, lines_cleared in scored_children:
            if depth == 1 or g_after.game_over:
                child_val = utility(g_after, root_player_id, lines_cleared)
            else:
                child_val = minimax_value(g_after, depth - 1, root_player_id, True, branch_limit, alpha, beta)

            value = min(value, child_val)
            beta = min(beta, value)
            if alpha >= beta:
                break 
        return value

def minimax_move(game, depth=3, branch_limit=8):
    """
    Top-level decision function.

    Returns the move-path (same format as ai.random_move) that maximizes
    the minimax value from the perspective of the *current* player.

    depth controls how many plies (half-moves) to look ahead.
    depth = 1  -> greedy, looks only at immediate outcome of this block.
    depth = 2+ -> also accounts for the opponent's response, etc.
    """
    root_player_id = game.current_player_id

    moves_dict = get_all_end_positions(game)
    if not moves_dict:
        return None

    # Prioritize immediate line clears
    best_lines = -1
    best_moves = []

    for item in moves_dict:
        path = item[1]
        g_copy = copy_game(game)
        g_after, lines_cleared = simulate_path(g_copy, path)

        if lines_cleared > best_lines:
            best_lines = lines_cleared
            best_moves = [(path, g_after, lines_cleared)]
        elif lines_cleared == best_lines:
            best_moves.append((path, g_after, lines_cleared))
    
    # If move clears 2+ lines, always choose it
    if best_lines >= 2:
        return random.choice([path for path, _, _ in best_moves])

    # If move clears 1 line,run minimax but heavily weight the value
    if best_lines == 1:
        line_value = []
        for path, g_after, lines in best_moves:
            val = minimax_value(g_after, depth - 1, root_player_id, False, branch_limit)
            line_value.append((val + 100, path))  
        line_value.sort(reverse=True)
        return line_value[0][1]
    
    # Otherwise, run minimax 
    children = []
    for item in moves_dict:
        path = item[1]
        g_copy = copy_game(game)
        g_after, lines_cleared = simulate_path(g_copy, path)
        val = utility(g_after, root_player_id, lines_cleared)
        children.append((val, path, g_after, lines_cleared))

    children.sort(key=lambda x: x[0], reverse=True)
    children = children[:branch_limit]

    best_value = -inf
    best_path = None
    for _, path, g_after, lines_cleared in children:
        val = minimax_value(g_after, depth - 1, root_player_id, False, branch_limit)
        if val > best_value:
            best_value = val
            best_path = path

    return best_path
