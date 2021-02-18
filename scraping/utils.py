import requests
from bs4 import BeautifulSoup
from time import sleep


def get_html(url):
    """
    Fetch the HTML source page from an url

    Parameters
    ----------
    url : String
        The url to the webpage

    Returns
    ------
    String:
        A BeautifulSoup object containing the HTML source page
    """

    while True:
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser', from_encoding='iso-8859-1')
            break
        except requests.exceptions.ConnectionError:
            sleep(1)

    return soup
