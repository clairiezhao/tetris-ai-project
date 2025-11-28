# ai_expectimax.py

import random
import copy

# We reuse the BFS-based move generator from the existing ai.py module.
# get_all_end_positions(game) returns a dict mapping final block states
# to a *path* (list of ints) describing how to move/rotate the current
# block to reach that state.
from ai import get_all_end_positions


def utility(game, root_player_id):
    """
    Utility function U = myScore - oppScore from the perspective of
    root_player_id (0 or 1).

    Assumes Game has game.player1 and game.player2, each with .score.
    """
    p1_score = game.player1.score
    p2_score = game.player2.score
    if root_player_id == 0:
        return p1_score - p2_score
    else:
        return p2_score - p1_score


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

    return game


def copy_game(game):
    """
    Create a deep copy of the Game state for search.

    This relies on the Game / Grid / Block / Player objects being
    reasonably deepcopy-able. If you run into issues, you may want
    to implement an explicit Game.clone() method and call that here.
    """
    return copy.deepcopy(game)


def expectimax_value(game, depth, root_player_id):
    """
    Recursive expectiminimax-style value function (MAX vs MIN, no explicit
    chance nodes here; the randomness comes from Game.get_random_block()).

    depth is measured in *plies* (half-moves).
    """
    # Terminal or depth limit: evaluate the state.
    if depth <= 0 or game.game_over:
        return utility(game, root_player_id)

    # Generate all possible end positions for the *current* block / player.
    moves_dict = get_all_end_positions(game)

    # If no legal moves (should be rare), just evaluate.
    if not moves_dict:
        return utility(game, root_player_id)

    values = []

    for path in moves_dict.values():
        # Work on a copy so we don't mutate the original game state.
        g_copy = copy_game(game)
        g_after = simulate_path(g_copy, path)
        v = expectimax_value(g_after, depth - 1, root_player_id)
        values.append(v)

    # If it is the root player to move, this is a MAX node; otherwise MIN.
    if game.current_player_id == root_player_id:
        return max(values)
    else:
        return min(values)


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

    best_value = None
    best_path = None

    for path in moves_dict.values():
        g_copy = copy_game(game)
        g_after = simulate_path(g_copy, path)
        value = expectimax_value(g_after, depth - 1, root_player_id)

        if (best_value is None) or (value > best_value):
            best_value = value
            best_path = path

    return best_path
