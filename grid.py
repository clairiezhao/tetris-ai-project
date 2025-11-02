import pygame;
from colors import Colors

class Grid:
    def __init__(self):
        self.num_rows = 20
        self.num_cols = 10
        self.cell_size = 30
        self.grid = [[0 for j in range(self.num_cols)] for i in range(self.num_rows)]
        self.colors = Colors.get_cell_colors()
    
    def print_grid(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                print(self.grid[row][col], end = " ")
            print()
    
    # is given cell inside grid?
    def is_inside(self, row, col):
        return row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols
    
    # is given cell empty (no block placed)
    def is_empty(self, row, col):
        return self.grid[row][col] == 0
    
    def is_row_full(self, row):
        for col in range(self.num_cols):
            if self.grid[row][col] == 0:
                return False
        return True
    
    def clear_row(self, row):
        for col in range(self.num_cols):
            self.grid[row][col] = 0
    
    # move a given row down by a certain number of rows
    # clears original row
    def move_row_down(self, row, num_rows):
        for col in range(self.num_cols):
            self.grid[row + num_rows][col] = self.grid[row][col]
            self.grid[row][col] = 0

    # scan all rows starting from bottom
    # if a row is full, increment num of completed rows, clear row
    # if not, move row down by num of completed rows
    # returns num of completed rows
    def clear_full_rows(self):
        completed = 0
        for row in range(self.num_rows - 1, 0, -1):
            if self.is_row_full(row):
                self.clear_row(row)
                completed += 1
            elif completed > 0:
                self.move_row_down(row, completed)
        return completed

    def reset(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.grid[row][col] = 0

    def draw(self, screen):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_val = self.grid[row][col]
                # define rectangle: x, y, width, height
                # offset of 1 to show grid lines
                cell_rect = pygame.Rect(col * self.cell_size + 11, row * self.cell_size + 11, self.cell_size - 1, self.cell_size - 1)
                # draw rect: surface to draw on, color, rectangle
                pygame.draw.rect(screen, self.colors[cell_val], cell_rect)