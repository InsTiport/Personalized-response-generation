import os
import tqdm
from interview_link_scrapper import get_player_interview_links_for_one_sport
from interview_text_scrapper import get_interview_text
from typing import Dict, List
import argparse


ID_LOOKUP = {
    'football': 1,
    'baseball': 2,
    'auto_racing': 3,
    'golf': 4,
    'hockey': 5,
    'tennis': 7,
    'equestrian': 9,
    'track_and_field': 10,
    'basketball': 11,
    'wrestling': 12,
    'boxing': 13,
    'soccer': 14,
    'extreme': 15,
    'cosida': 17,
    'volleyball': 19,
    'lacrosse': 20,
    'swimming': 21,
    'cricket': 22
}


def main():
    # setup args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-s', '--sport',
        default='football',
        choices=list(ID_LOOKUP.keys()),
        help=f'Specify the sport type)'
    )
    args = arg_parser.parse_args()

    sport_type = args.sport

    # file system routine
    os.chdir('../')
    sport_folder_path = os.path.join('data', sport_type)

    sport_url = f'http://www.asapsports.com/showcat.php?id={ID_LOOKUP[sport_type]}&event=yes'

    print(f'Will store scraped interviews in {sport_folder_path}/')
    os.makedirs(os.path.dirname(sport_folder_path + '/'), exist_ok=True)

    # get all interviews for all players
    print(f'Getting all interviews for all {sport_type} players...')
    player_interview_links: Dict[str, List[str]] = get_player_interview_links_for_one_sport(sport_url)

    # write interviews to text files
    print(f'Writing interviews for all {sport_type} players to files...')
    # keep track of some urls that are not able to decipher
    exclude = set()
    for player, player_interview_urls in tqdm.tqdm(player_interview_links.items()):
        os.makedirs(os.path.dirname(os.path.join(sport_folder_path, player) + '/'), exist_ok=True)
        count = 1
        for interview_url in player_interview_urls:
            correct, name, time, players, text = get_interview_text(interview_url)
            if correct:
                filename = os.path.join(sport_folder_path, player, str(count))
                count += 1
                with open(filename, 'w') as f:
                    f.write(name + '\n')
                    f.write(time + '\n')
                    for player_name in players:
                        f.write(player_name + '\n')
                    f.write('START_OF_INTERVIEW_TEXT' + '\n')
                    f.write(text + '\n')
                    f.write('END_OF_INTERVIEW_TEXT')
                    f.close()
            else:
                exclude.add(interview_url)

    # record urls that are not decoded correctly
    with open(os.path.join(sport_folder_path, 'excluded_url.txt'), 'w') as f:
        for excluded_url in exclude:
            f.write(excluded_url + '\n')


if __name__ == '__main__':
    main()
