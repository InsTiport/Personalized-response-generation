# Personalized response generation

## Web scraping
All scraped interview transcripts are located in `data` folder. Football and basketball interviews have already been scraped.

To scrape interviews of other type of sports, inside `scraping` folder, run:

```python scraper.py -s [sport name]```

Use `-h` to see available sports.

## Generating `*_utterance.csv` and `*_episode.csv`
These files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). 

To generate these files on your local machine, inside `processing` folder, run:

```python generate_csv.py```

You can find these files in `data/csv` after they are generated. Currently, only football and basketball interviews are available, and you should see 30,974 episodes and 4,442,689 utterances in total.

## Troubleshooting
If you see a `ModuleNotFoundError` while running any script on command line, this is because local scripts are not added to `PYTHONPATH`. Please use an IDE which has the functionality of automatically adding content roots to `PYTHONPATH` to run that script.
