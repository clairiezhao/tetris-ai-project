import pygame
import sys
from game import Game
from colors import Colors
import ai

pygame.init()

title_font = pygame.font.Font(None, 40)
score1_bg_surface = title_font.render("Player 1", True, Colors.white)
score2_bg_surface = title_font.render("Player 2", True, Colors.white)
next_bg_surface = title_font.render("Next", True, Colors.white)
game_over_surface = title_font.render("GAME OVER", True, Colors.white)
score1_rect = pygame.Rect(320, 55, 170, 60)
score2_rect = pygame.Rect(320, 165, 170, 60)
next_rect = pygame.Rect(320, 315, 170, 180)


# game window: 300w x 600h pixels
# offset of 200, 20
# (0,0) is top left corner
screen = pygame.display.set_mode((500, 620))
pygame.display.set_caption("Tetris")
clock = pygame.time.Clock()
game = Game()

# USEREVENT = custom event
GAME_UPDATE = pygame.USEREVENT
# trigger game update event every 200 ms
pygame.time.set_timer(GAME_UPDATE, 200)

# timer for AI moves
AI_MOVE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(AI_MOVE_EVENT, 200)

ai_path = []
ai_moving = False

# game loop: event handling, update positions, draw objects
while True:
    last_player_id = game.current_player_id

    for event in pygame.event.get():
        # quit event: click x button on window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if game.game_over == True:
                    game.game_over = False
                    game.reset()
            # player 1 uses arrow keys
            if(game.current_player_id == 0 and (not game.game_over)):
                if event.key == pygame.K_LEFT:
                    game.move_left()
                if event.key == pygame.K_RIGHT:
                    game.move_right()
                if event.key == pygame.K_DOWN:
                    game.move_down()
                    game.update_score(game.current_player_id, 0, 1)
                if event.key == pygame.K_UP:
                    game.rotate()
            # player 2 uses a(=LFT), w(=UP), s(=DWN), d=(RGT)
            # elif((game.current_player_id == 1) and (not game.game_over)):
            #     if event.key == pygame.K_a :
            #         game.move_left()
            #     if event.key == pygame.K_d:
            #         game.move_right()
            #     if event.key == pygame.K_s:
            #         game.move_down()
            #         game.update_score(game.current_player_id, 0, 1)
            #     if event.key == pygame.K_w:
            #         game.rotate()
        if event.type == GAME_UPDATE and (not game.game_over):
            game.move_down()

            if ai_path:
                move = ai_path.pop(0)
                
                if move == 0:
                    game.move_down()
                elif move == 1:
                    game.move_left()
                elif move == 2:
                    game.move_right()
                else:
                    game.rotate()
            # else:
            #     game.move_down()
    
    if game.current_player_id != last_player_id:
        if game.current_player_id == 1:
            ai_path = ai.random_move(game)
            ai_moving = True

            if not ai_path:
                game.game_over = True
        else:
            ai_path = []
            ai_moving = False

    # ui graphics
    score1_surface = title_font.render(str(game.player1.score), True, Colors.white)
    score2_surface = title_font.render(str(game.player2.score), True, Colors.white)
    screen.fill(Colors.dark_blue)
    screen.blit(score1_bg_surface, (350, 20, 50, 50))
    screen.blit(score2_bg_surface, (350, 130, 50, 50))
    screen.blit(next_bg_surface, (375, 280, 50, 50))
    if game.game_over:
        screen.blit(game_over_surface, (320, 500, 50, 50))

    # scoreboards
    pygame.draw.rect(screen, Colors.light_blue, score1_rect, 0, 10)
    screen.blit(score1_surface, score1_surface.get_rect(centerx = score1_rect.centerx, centery = score1_rect.centery))
    pygame.draw.rect(screen, Colors.light_blue, score2_rect, 0, 10)
    screen.blit(score2_surface, score2_surface.get_rect(centerx = score2_rect.centerx, centery = score2_rect.centery))

    # next block
    pygame.draw.rect(screen, Colors.light_blue, next_rect, 0, 10)
    game.draw(screen)

    pygame.display.update()
    # fps = 60
    clock.tick(60)