from grid import Grid
from blocks import *
from player import Player
import random

class Game:
    def __init__(self):
        self.grid = Grid()
        self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        self.current_block = self.get_random_block()
        self.next_block = self.get_random_block()
        self.game_over = False
        self.score = 0

        # multiplayer attr
        self.num_players = 2
        self.player1 = Player(0)
        self.player2 = Player(1)
        # start from player 1
        self.current_player_id = 0
        self.end_turn = False

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
            self.grid.grid[pos.row][pos.col] = self.current_block.id
        self.current_block = self.next_block
        self.next_block = self.get_random_block()
        # when a block is locked in place, clear all completed rows
        rows_cleared = self.grid.clear_full_rows()
        self.update_score(self.current_player_id, rows_cleared, 0)
        if not self.block_fits(self.current_block):
            self.game_over = True
        # when a block is locked in place, current player's turn ends
        self.change_players()

    # check if a block does not collide with any existing block
    def block_fits(self, block):
        tiles = block.get_cell_positions()
        for pos in tiles:
            if not self.grid.is_inside(pos.row, pos.col):
                return False
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
        self.current_block.draw(screen, 11, 11)

        if self.next_block.id == 3:
            self.next_block.draw(screen, 255, 390)
        elif self.next_block.id == 4:
            self.next_block.draw(screen, 255, 380)
        else:
            self.next_block.draw(screen, 270, 370)