import os
import tqdm
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.espn_text_scraper import get_report_text
from scraping.espn_link_scraper import get_reports_link_for_one_sport

root_archive_links = [
    'http://www.espn.com/college-football/news/archive',
    'http://www.espn.com/nfl/news/archive',
    'http://www.espn.com/mens-college-basketball/news/archive',
    'https://www.espn.com/mlb/news/archive'
]

def main():
    os.chdir('../')
    os.makedirs(os.path.join('data', 'espn'), exist_ok=True)

    report_id = 0
    for archive_link in root_archive_links:
        report_links_found = get_reports_link_for_one_sport(archive_link)
        for report_link in report_links_found:
            print(report_link)
            text = get_report_text(report_link)
            if text != "":
                with open(os.path.join('data', 'espn', str(report_id)), 'w') as f:
                    f.write(text)
                report_id += 1
                print(text)

if __name__ == '__main__':
    main()