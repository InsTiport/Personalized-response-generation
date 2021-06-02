import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from scraping.utils import get_html

def get_report_text(report_url):
    soup = get_html(report_url)
    if soup.title == None:
        return ""
    title = soup.title.string
    body = ""
    for p in soup.find_all('p'):
        body += p.get_text()
    
    return title + "\n" + body

if __name__ == '__main__':
    print(get_report_text('https://www.espn.com/college-sports/story/_/id/29364287/john-swofford-retire-acc-commissioner-2020-21-campaign')[1])
