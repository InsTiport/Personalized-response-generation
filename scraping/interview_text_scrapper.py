from bs4 import NavigableString, Tag
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.utils import get_html


def get_interview_text(interview_url):
    """
    Fetch a single piece of interview text and meta-data from a source webpage

    Parameters
    ----------
    interview_url : String
        The url to the webpage

    Returns
    ------
    correct : boolean
        whether this interview text has been decoded correctly

    interview_name : String
        Name of this interview

    interview_time : String
        When this interview happened

    interview_players: List[String]
        Interviewees

    interview_text : String
        An unprocessed String of raw interview text (including Questions and interviewee responses)
    """
    # example url: http://www.asapsports.com/show_conference.php?id=144725

    # fetch HTML
    soup = get_html(interview_url)

    if len(soup.find_all('h1')) != 1:
        print('h1 Tag is not unique, check url for details:')
        print(interview_url)
    if soup.find_all('h1')[0].a is not None:
        interview_name = str(soup.find_all('h1')[0].a.contents[0])
    else:
        interview_name = str(soup.find_all('h1')[0].contents[0])
    if len(soup.find_all('h2')) != 1:
        print('h2 Tag is not unique, check url for details:')
        print(interview_url)
    interview_time = str(soup.find_all('h2')[0].contents[0])

    # find all players attending this interview
    interview_players = []
    for link in soup.find_all('a'):
        if link.get('href') is not None and 'show_player.php' in link.get('href'):
            interview_players.append(str(link.contents[0]))

    # find interview text
    for td in soup.find_all('td'):
        if td.get('valign') == 'top' and td.get('style') == 'padding: 10px;':
            raw_interview_text = td.contents
            interview_text = ''
            for item in raw_interview_text:
                '''
                 all actual texts are either directly below the td Tag or is a Tag with name 'b' or 'p'
                 sometimes, texts are under strong Tags, especially for cricket
                '''
                # CASE 1: directly below td
                if type(item) is NavigableString:
                    interview_text += ' ' + str(item)
                # CASE 2: 'b' or 'p' Tag
                elif type(item) is Tag and (item.name == 'b' or item.name == 'p' or item.name == 'br'):
                    interview_text += ' ' + item.text
                elif type(item) is Tag and item.name == 'strong':
                    if item.text != 'FastScripts Transcript by ASAP Sports':
                        interview_text += ' ' + item.text

    # check whether this interview has been processed correctly
    correct = True if (len(interview_text) > 100 and '<' not in interview_text) else False
    # if not correct:
    #     print(interview_url)

    # remove #nbsp; \t and Â from text
    interview_text = interview_text.replace('\xa0', ' ')
    interview_text = interview_text.replace('\t', '\n')
    interview_text = interview_text.replace('Â', ' ')
    # sometimes, this line will also be included :(
    interview_text = interview_text.replace('FastScripts Transcript by ASAP Sports', '')

    return correct, interview_name, interview_time, interview_players, interview_text


def process_interview_text(text):
    """
    Process a raw interview text by separating questions from interviewee responses

    Parameters
    ----------
    text : String
        An unprocessed String of raw interview text (including Questions and interviewee responses)

    Returns
    ------
    String
        An processed String of raw interview text (including Questions and interviewee responses)
    """
    pass
