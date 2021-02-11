from typing import Dict, List
import tqdm
from utils import get_html


def get_player_interview_links_for_one_sport(sport_url):
    """
    Fetch all interview links associated with all players for one sport

    Parameters
    ----------
    sport_url : String
        The url to the webpage which is the entry page to one sport

    Returns
    ------
    Dictionary
        A dictionary of pairs <player_name, list of interview urls for this player> for this sport
    """

    '''
    get each initial's link (26 links, for names starting with A-Z)
    '''
    soup = get_html(sport_url)
    player_initial_links = []
    for link in soup.find_all('a'):
        if 'letter=' in link.get('href'):
            player_initial_links.append(link.get('href'))

    '''
    get each player's link (one link per player)
    '''
    player_links: Dict[str, str] = dict()  # <player_name, player_url>
    for initial_link in player_initial_links:
        # get the webpage which contains all players whose names start with this initial
        soup = get_html(initial_link)

        # find all players on this page
        for link in soup.find_all('a'):
            if 'show_player.php' in link.get('href'):
                player_name = link.contents[0]
                player_links[player_name] = link.get('href')

    '''
    get all interviews for each player
    '''
    player_interview_links: Dict[str, List[str]] = dict()  # <player_name, list of interview urls for this player>
    for player, player_link in tqdm.tqdm(player_links.items()):
        # initialize list to store all interviews for this player
        player_interview_links[player] = []
        # get the webpage which contains all interviews of this player
        soup = get_html(player_link)

        # find all interviews for this player
        for interview_link in soup.find_all('a'):
            if 'show_interview.php' in interview_link.get('href') or 'show_conference.php' in interview_link.get(
                    'href'):
                player_interview_links[player].append(interview_link.get('href'))

    return player_interview_links

