import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.utils import get_html


def get_reports_link_for_one_sport(sport_url):
    years = range(2003, 2004, 1)
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
    links = []
    for year in years:
        # print(f"{year}")
        for month in months:
            # print(f"\t{month}")
            soup = get_html(sport_url + f'?month={month}&year={year}')
            for link in soup.find_all('li'):
                if str(year) in link.get_text():
                    links.append(link.a.get('href'))
    return links


if __name__ == '__main__':
    get_reports_link_for_one_sport('http://www.espn.com/college-football/news/archive')