from grid import Grid
from blocks import *
from player import Player
from colors import Colors
import random

class Game:
    def __init__(self):
        self.grid = Grid()
        self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        self.current_block = self.get_random_block()
        self.next_block = self.get_random_block()
        self.game_over = False
        self.board_full = False
        self.score = 0

        # multiplayer attr
        self.num_players = 2
        self.player1 = Player(0, Colors.red)
        self.player2 = Player(1, Colors.blue)
        self.player1.other_player = self.player2
        self.player2.other_player = self.player1

        # start from player 1
        self.current_player_id = 0
        self.end_turn = False

    # update score for either player based on # of lines cleared and # of move down points
    def update_score(self, player_id, lines_cleared, move_down_points):
        if player_id == 0:
            self.player1.update_score(lines_cleared, move_down_points)
        elif player_id == 1:
            self.player2.update_score(lines_cleared, move_down_points)
    
    # selects random block according to 7 bag system
    # each block appears at least once in a cycle
    def get_random_block(self):
        if len(self.blocks) == 0:
            self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        block = random.choice(self.blocks)
        self.blocks.remove(block)
        return block
    
    def move_left(self):
        self.current_block.move(0, -1)
        if not self.block_fits(self.current_block):
            self.current_block.move(0, 1)
    
    def move_right(self):
        self.current_block.move(0, 1)
        if not self.block_fits(self.current_block):
            self.current_block.move(0, -1)
    
    def move_down(self):
        self.current_block.move(1, 0)
        # if block does not fit completely inside board or block collides with other block
        if not self.block_fits(self.current_block):
            self.current_block.move(-1, 0)
            self.lock_block()

    # lock block colors in place
    # generate next block
    def lock_block(self):
        tiles = self.current_block.get_cell_positions()
        for pos in tiles: 
            self.grid.grid[pos.row][pos.col] = self.current_player_id + 1
        self.current_block = self.next_block
        self.next_block = self.get_random_block()
        # when a block is locked in place, clear all completed rows
        rows_cleared = self.grid.clear_full_rows()
        self.update_score(self.current_player_id, rows_cleared, 0)
        if not self.block_fits(self.current_block):
            self.board_full = True
        # when a block is locked in place, current player's turn ends
        self.change_players()

    # check if a block does not collide with any existing block
    def block_fits(self, block):
        tiles = block.get_cell_positions()
        for pos in tiles:
            # check if tile is within bounds of board
            if not self.grid.is_inside(pos.row, pos.col):
                return False
            # checks that all slots are empty
            if not self.grid.is_empty(pos.row, pos.col):
                return False
        return True
    
    def rotate(self):
        rotate_curr = self.current_block.clone()
        rotate_curr.rotate()
        if self.block_fits(rotate_curr):
            self.current_block.rotate()
    
    def reset(self):
        self.grid.reset()
        self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        self.current_block = self.get_random_block()
        self.next_block = self.get_random_block()
        self.score = 0
    
    def change_players(self):
        self.current_player_id = (self.current_player_id + 1) % self.num_players

    def draw(self, screen):
        self.grid.draw(screen)
        if self.current_player_id == 0:
            curr_color = self.player1.color
            next_color = self.player2.color
        else:
            curr_color = self.player2.color
            next_color = self.player1.color
        self.current_block.draw(screen, 11, 11, curr_color)

        if self.next_block.id == 3:
            self.next_block.draw(screen, 255, 490, next_color)
        elif self.next_block.id == 4:
            self.next_block.draw(screen, 255, 480, next_color)
        else:
            self.next_block.draw(screen, 270, 470, next_color)
    
    def clear_board(self):
        self.grid.reset()
        self.board_full = False
        
    def clone(self):
        """
        Create a full game-state clone safe for AI search.
        Only copies logical state; no pygame/display objects involved.
        """
        # 1. Create a new Game instance without calling __init__
        g = Game.__new__(Game)

        # 2. Clone grid
        g.grid = Grid()
        g.grid.grid = [row[:] for row in self.grid.grid]

        # 3. Clone remaining bag of blocks (7-bag)
        #    Each element in self.blocks is an IBlock/JBlock/etc. instance.
        #    Those classes already support .clone() because rotate() uses clone().
        g.blocks = [b.clone() for b in self.blocks]

        # 4. Clone current and next blocks
        g.current_block = self.current_block.clone() if self.current_block is not None else None
        g.next_block = self.next_block.clone() if self.next_block is not None else None

        # 5. Copy basic game flags / fields
        g.game_over = self.game_over
        g.score = self.score

        # 6. Multiplayer attributes
        g.num_players = self.num_players

        g.player1 = Player(0, Colors.red)
        g.player1.score = self.player1.score

        g.player2 = Player(1, Colors.blue)
        g.player2.score = self.player2.score

        # Re-link the cross references
        g.player1.other_player = g.player2
        g.player2.other_player = g.player1

        g.current_player_id = self.current_player_id
        g.end_turn = self.end_turn

        return g
