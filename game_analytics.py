import os

# Specifications. Change based on possible sport names (see scraper.py)
####################################################################
sport_type = 'football'
####################################################################

SPORT_FOLDER_PATH = os.path.join('data', sport_type)

games_list = set()
for player_folder in os.scandir(SPORT_FOLDER_PATH):
    for interview_text in os.scandir(player_folder):
        with open(interview_text) as f:
            game_title = f.readline().strip()
            games_list.add(game_title)

game_types_list = set()
for game in games_list:
    game_types_list.add(game.split(':')[0])

print(f"There are {len(games_list)} distinct {sport_type} games in total.")
print(f"There are {len(game_types_list)} distinct {sport_type} game types in total")