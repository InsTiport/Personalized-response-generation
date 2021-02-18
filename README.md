# Personalized response generation

## Web scraping
All scraped interview transcripts are located in `data` folder. Football and basketball interviews have already been scraped.

To scrape interviews of other type of sports, inside `scraping` folder, run:

```python scraper.py -s [sport name]```

Use `-h` to see available sports.

## Generating `utterance.csv` and `episode.csv`
These two files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). To generate these two files on your local machine, inside `processing` folder, run:

```python generate_csv.py```

Currently, only football interviews will be processed to generate these two files.
