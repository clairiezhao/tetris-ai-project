import pygame
import sys
from game import Game
from colors import Colors
import ai
import copy

pygame.init()

# simple ui since don't need scores or anything
screen = pygame.display.set_mode((500, 620))
clock = pygame.time.Clock()
game = Game()

# speed up timer
GAME_UPDATE = pygame.USEREVENT
pygame.time.set_timer(GAME_UPDATE, 50) 

ai_path = []
ai_moving = False

moves = []
curr_move_ind = 0
curr_shape = game.current_block.clone()

game.current_player_id = 0
game.num_players = 2 

def get_all_moves():
    global moves, curr_move_ind, curr_shape, ai_path, ai_moving

    game.current_block = game.get_random_block()
    curr_shape = game.current_block.clone()
    
    moves = list(ai.get_all_end_positions(game).values())
    curr_move_ind = 0

    if not moves:
        get_all_moves()
        return
    else:
        print(f"{len(moves)} moves for this block")
        ai_path = copy.deepcopy(moves[curr_move_ind])
        ai_moving = True

get_all_moves()

while True:
    last_player_id = game.current_player_id

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == GAME_UPDATE:
            game.move_down()
 
            if ai_moving and ai_path:
                move = ai_path.pop(0)
                
                if move == 0:
                    game.move_down()
                elif move == 1:
                    game.move_left()
                elif move == 2:
                    game.move_right()
                elif move == 3:
                    game.rotate()
    

    if game.current_player_id != last_player_id:
        
        screen.fill(Colors.dark_blue)
        game.draw(screen)
        pygame.display.update()
        pygame.time.wait(100)

        curr_move_ind += 1
        if curr_move_ind >= len(moves):
            get_all_moves()
        else:
            game.grid.reset()
            game.current_block = curr_shape.clone()
            ai_path = copy.deepcopy(moves[curr_move_ind])
            ai_moving = True
        
        # always ai's turn
        game.current_player_id = 0
        
        if not ai_path:
            game.grid.reset()
            ai_path = ai.random_move(game)
            if not ai_path:
                game.game_over = True

    if not game.game_over:
        screen.fill(Colors.dark_blue)
        game.draw(screen)
        pygame.display.update()

    clock.tick(60)