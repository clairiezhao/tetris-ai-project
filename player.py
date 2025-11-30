class Player:
    def __init__(self, id, color):
        # 0 or 1 for two player
        self.id = id
        self.score = 0
        self.other_player = None
        # red for 0, blue for 1
        self.color = color

    def update_score(self, lines_cleared, move_down_points):
        if lines_cleared == 1:
            self.score += 40
        elif lines_cleared == 2:
            self.score += 100
        elif lines_cleared == 3:
            self.score += 300
        elif lines_cleared == 4:
            self.score += 1200
        self.score += move_down_points
    

        