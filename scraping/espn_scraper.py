import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.espn_link_scraper import get_reports_link_for_one_sport

root_archive_links = [
    # football
    ('http://www.espn.com/college-football/news/archive', 'football'),
    ('http://www.espn.com/nfl/news/archive', 'football'),
    # basketball
    ('http://www.espn.com/mens-college-basketball/news/archive', 'basketball'),
    ('http://www.espn.com/womens-college-basketball/news/archive', 'basketball'),
    ('http://www.espn.com/nba/news/archive', 'basketball'),
    # baseball
    ('https://www.espn.com/mlb/news/archive', 'baseball'),
    # golf
    ('https://www.espn.com/golf/news/archive', 'golf'),
    # hockey
    ('https://www.espn.com/nhl/news/archive', 'hockey')
]


def main():
    os.chdir('../')
    os.makedirs(os.path.join('data', 'espn'), exist_ok=True)

    for archive_link, game_dir_name in root_archive_links:
        get_reports_link_for_one_sport(archive_link, game_dir_name)


if __name__ == '__main__':
    main()
