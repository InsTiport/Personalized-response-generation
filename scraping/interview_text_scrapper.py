from bs4 import NavigableString, Tag
from utils import get_html


def get_interview_text(interview_url):
    """
    Fetch a single piece of interview text and meta-data from a source webpage

    Parameters
    ----------
    interview_url : String
        The url to the webpage

    Returns
    ------
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

    assert len(soup.find_all('h1')) == 1
    if soup.find_all('h1')[0].a is not None:
        interview_name = str(soup.find_all('h1')[0].a.contents[0])
    else:
        interview_name = str(soup.find_all('h1')[0].contents[0])
    assert len(soup.find_all('h2')) == 1
    interview_time = str(soup.find_all('h2')[0].contents[0])

    # find all players attending this interview
    interview_players = []
    for link in soup.find_all('a'):
        if 'show_player.php' in link.get('href'):
            interview_players.append(str(link.contents[0]))

    # find interview text
    for td in soup.find_all('td'):
        if td.get('valign') == 'top' and td.get('style') == 'padding: 10px;':
            raw_interview_text = td.contents
            interview_text = ''
            for item in raw_interview_text:
                '''
                 all actual texts are either directly below the td Tag or is a Tag with name 'b' or 'p'
                '''
                # CASE 1: directly below td
                if type(item) is NavigableString:
                    interview_text += str(item)
                # CASE 2: 'b' Tag
                elif type(item) is Tag and item.name == 'b':
                    # cope with empty tags: <b></b>
                    if len(item.contents) > 0:
                        # there may be a 'p' Tag under a 'b' Tag
                        for sub_item in item:
                            if type(sub_item) is NavigableString:
                                interview_text += str(sub_item)
                            elif type(sub_item) is Tag and sub_item.name == 'p':
                                # potential empty p Tag?
                                if len(sub_item.contents) > 0:
                                    interview_text += str(sub_item.contents[0])
                # CASE 3: 'p' Tag
                elif type(item) is Tag and item.name == 'p':
                    if len(item.contents) > 0:
                        interview_text += str(item.contents[0])

    # remove #nbsp; \t and Â from text
    interview_text = interview_text.replace('\xa0', ' ')
    interview_text = interview_text.replace('\t', '\n')
    interview_text = interview_text.replace('Â', ' ')

    return interview_name, interview_time, interview_players, interview_text


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
