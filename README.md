# Personalized response generation

## Quickstart
inside `processing`, run the following commands in order:

```python generate_csv.py```

```python generate_indexed_csv.py```

This will generate all (`*_utterance_indexed.csv`) files to `data`.

## Documentation
### Web scraping
All scraped interview transcripts are located in `data` folder. Football and basketball interviews have already been scraped.

To scrape interviews of other type of sports, inside `scraping` folder, run:

```python scraper.py -s [sport name]```

Use `-h` to see available sports.

### Generate `*_utterance.csv` and `*_episode.csv`
These files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). You must run the following command to generate these files. 

To generate these files on your local machine, inside `processing` folder, run:

```python generate_csv.py```

You can find these files in `data/csv` after they are generated. Currently, only football and basketball interviews are available, and you should see 30,974 episodes and 4,442,689 utterances in total.

### Build vocabulary
No need to perform this step again because the vocabulary files are already generated and pushed to Github, under `data/vocab`.

If there are new interview texts available and you need to rebuild the vocabulary, first re-generate all csv files by following the instruction above, then inside `processing` folder, run:

```python build_vocab.py```

This will generate the vocabulary files again, and remove the older vocabulary files.

### Generate indexed versions of `*_utterance.csv` (`*_utterance_indexed.csv`)
These files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). You must run the following command to generate these files.

Before running the following command, make sure you have already generated `*_utterance.csv` and `*_episode.csv`. To generate these files on your local machine, inside `processing` folder, run:

```python generate_indexed_csv.py```

## Troubleshooting
If you see a `ModuleNotFoundError` while running any script on command line, this is because local scripts are not added to `PYTHONPATH`. Please use an IDE which has the functionality of automatically adding content roots to `PYTHONPATH` to run that script.
