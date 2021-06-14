import os
import sys
from tqdm import tqdm
sys.path.insert(0, os.path.abspath('..'))
from scraping.utils import get_html
from scraping.espn_text_scraper import get_report_text


def get_reports_link_for_one_sport(sport_url, game_dir_name):
    years = range(2003, 2022)
    months = [
        'january',
        'february', 
        'march', 
        'april', 
        'may', 
        'june', 
        'july', 
        'august', 
        'september', 
        'october', 
        'november', 
        'december'
    ]

    print(f"Scraping links in {sport_url}")
    for year in years:
        print(f"{year}")
        for month in tqdm(months):
            # print(f"\t{month}")
            soup = get_html(sport_url + f'?month={month}&year={year}')
            for link in soup.find_all('li'):
                if str(year) in link.get_text():
                    try:
                        text = get_report_text(link.a.get('href'))
                        if text != "":
                            time = str(link.contents[1])
                            date = time[:time.index(',')].split()[-1]

                            text = text.replace('ESPN.com: ', '').replace('/', ':')
                            title = text[:text.index('\n')]
                            os.makedirs(os.path.join('data', 'espn', game_dir_name, str(year), month, date),
                                        exist_ok=True)
                            with open(os.path.join('data', 'espn', game_dir_name, str(year), month, date, title), 'w')\
                                    as f:
                                f.write(text)
                    except Exception:
                        continue


if __name__ == '__main__':
    get_reports_link_for_one_sport('http://www.espn.com/college-football/news/archive')