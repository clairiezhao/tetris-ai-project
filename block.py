from colors import Colors
import pygame
from position import Position
import copy

class Block:
    def __init__(self, id):
        # block id
        self.id = id
        # dict: store occupied cells in bounding grid
        # keys = rotation state (1 to 4)
        self.cells = {}
        self.cell_size = 30
        # position offset to move block
        self.row_offset = 0
        self.col_offset = 0
        # 0 to 3
        self.rotation_state = 0
        self.colors = Colors.get_cell_colors()
    
    def clone(self):
        new_block = self.__class__()

        new_block.row_offset = self.row_offset
        new_block.col_offset = self.col_offset
        new_block.rotation_state = self.rotation_state
        new_block.id = self.id
        new_block.cells = self.cells

        return new_block

    def move(self, rows, cols):
        self.row_offset += rows
        self.col_offset += cols
    
    # ret new positions with applied offset
    def get_cell_positions(self):
        tiles = self.cells[self.rotation_state]
        moved_tiles = []
        for pos in tiles:
            pos = Position(pos.row + self.row_offset, pos.col + self.col_offset)
            moved_tiles.append(pos)
        return moved_tiles
    
    def rotate(self):
        self.rotation_state += 1
        if self.rotation_state == len(self.cells):
            self.rotation_state = 0

    def undo_rotate(self):
        self.rotation_state -= 1
        if self.rotation_state == 0:
            self.rotation_state = len(self.cells) - 1

    def draw(self, screen, offset_x, offset_y):
        # pos of each occupied cell
        tiles = self.get_cell_positions()
        for tile in tiles:
            tile_rect = pygame.Rect(offset_x + tile.col * self.cell_size, offset_y + tile.row * self.cell_size, self.cell_size - 1, self.cell_size - 1)
            pygame.draw.rect(screen, self.colors[self.id], tile_rect)
