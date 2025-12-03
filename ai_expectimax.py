# ai_expectimax.py

import random
import copy

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
        10000 * lines_cleared
        - 0.5 * agg_height
        - 0.7 * holes
        - 0.3 * bumpiness
        - 0.2 * max_height
    )


def utility(game, root_player_id, lines_cleared):
    """
    U = (myScore - oppScore) + board heuristic.
    The score term dominates; board heuristic breaks ties / guides early game.
    """
    p1_score = game.player1.score
    p2_score = game.player2.score
    if root_player_id == 0:
        score_diff = p1_score - p2_score
    else:
        score_diff = p2_score - p1_score

    board_val = board_heuristic(game.grid, lines_cleared)

    # Make score difference matter a lot, but still care about board shape
    return 100 * score_diff + board_val



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


def expectimax_value(game, depth, root_player_id, branch_limit=6):
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
        scored_children.append((score, path, lines_cleared))

    # Sort descending for MAX, ascending for MIN, then cut to branch_limit
    max_turn = (game.current_player_id == root_player_id)
    scored_children.sort(key=lambda x: x[0], reverse=max_turn)
    scored_children = scored_children[:branch_limit]

    # values = []
    # for _, path, lines_cleared in scored_children:
    #     g_copy = copy_game(game)
    #     g_after, next_lines = simulate_path(g_copy, path)
    #     v = expectimax_value(g_after, depth - 1, root_player_id, branch_limit)
    #     if max_turn:
    #         v += 10000 * lines_cleared
    #     values.append(v)

    # if max_turn:
    #     return max(values)
    # else:
    #     return min(values)

    if max_turn:
        best_value = None
        for _, path, lines_cleared in scored_children:
            g_copy = copy_game(game)
            g_after, _ = simulate_path(g_copy, path)
            v = expectimax_value(g_after, depth - 1, root_player_id, branch_limit)
            v += 10000 * lines_cleared
            best_value = max(best_value, v)
        return best_value
    else:
        total_value = 0
        for _, path, _ in scored_children:
            g_copy = copy_game(game)
            g_after, _ = simulate_path(g_copy, path)
            v = expectimax_value(g_after, depth - 1, root_player_id, branch_limit)
            total_value += v
        return total_value / max(len(scored_children), 1)


def expectimax_move(game, depth=2):
    """
    Top-level decision function.

    Returns the move-path (same format as ai.random_move) that maximizes
    the expectimax value from the perspective of the *current* player.

    depth controls how many plies (half-moves) to look ahead.
    depth = 1  -> greedy, looks only at immediate outcome of this block.
    depth = 2+ -> also accounts for the opponent's response, etc.
    """
    root_player_id = game.current_player_id

    moves_dict = get_all_end_positions(game)
    if not moves_dict:
        return None
    
    best_moves = []
    for item in moves_dict:
        path = item[1]
        g_copy = copy_game(game)
        # path is a list of moves - 0,1,2,3. See above for mapping of numbers to move
        g_after, lines_cleared = simulate_path(g_copy, path)
        if lines_cleared > 0:
            best_moves.append((lines_cleared, path))

    if best_moves:
        # Pick the move that clears the most lines
        return max(best_moves, key=lambda x: x[0])[1]

    best_value = None
    best_path = None

    for item in moves_dict:
        path = item[1]
        g_copy = copy_game(game)
        g_after, lines_cleared = simulate_path(g_copy, path)
        
        value = expectimax_value(g_after, depth - 1, root_player_id)

        if (best_value is None) or (value > best_value):
            best_value = value
            best_path = path

    return best_path
