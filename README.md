# Personalized response generation

## Quickstart
Get indexed version of tokenized texts [here](https://drive.google.com/drive/folders/1EzdSebTBt30p6iVQvq_3v5CURFqvkn6U?usp=sharing).

Alternatively, you can run the following commands to generate these files. Inside `processing`, run the following commands in order:

```python generate_csv.py```

```python generate_indexed_csv.py```

This will generate all (`*_utterance_indexed.csv`) files to `data`.

Note that some interview texts are non-English, and this phenomenon is especially prevalent among baseball, golf and soccer interviews. Right now, English language checking is performed while generating csv files (`generate_csv.py`), not when generating vocabularies. The current vocabulary size is 161,292.

## Documentation
### Web scraping
All scraped interview transcripts are located in `data/[sport_name]` folder. You don't need to redo this step since all interviews have been scrpaed.

To scrape interviews of other type of sports, inside `scraping` folder, run (use `-h` to see available sports):

```python scraper.py -s [sport_name]```

The scraper will also create an `excluded_url.txt` file in each `data/[sport_name]` folder. This file documents all urls that are excluded while processing because their contents are unable to decode properly for any reason.

### Generate `*_utterance.csv` and `*_episode.csv`
These files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). You must run the following command to generate these files. 

To generate these files on your local machine, inside `processing` folder, run:

```python generate_csv.py```

You can find these files in `data/csv` after they are generated. English language checking is performed in this step and `non_English_interviews.txt` will be generated in each `data/[sport_name]` folder. You should see 149,591 interviews and 15,639,895 utterances in total.

### Build vocabulary
No need to perform this step again because the vocabulary files are already generated and pushed to Github, under `data/vocab`. The current vocabulary size is 161,292.

If there are new interview texts available and you need to rebuild the vocabulary, first re-generate all csv files by following the instruction above, then inside `processing` folder, run:

```python build_vocab.py```

This will generate the vocabulary files again, and remove the older vocabulary files.

### Generate indexed versions of `*_utterance.csv` (`*_utterance_indexed.csv`)
These files are not on Github and should not be put under version control (they are too large), and Git should ignore all csv files (see `.gitignore`). You must run the following command to generate these files.

Before running the following command, make sure you have already generated `*_utterance.csv` and `*_episode.csv`. To generate these files on your local machine, inside `processing` folder, run:

```python generate_indexed_csv.py```

## Troubleshooting
If you see a `ModuleNotFoundError` while running any script on command line, this is because local scripts are not added to `PYTHONPATH`. Please use an IDE which has the functionality of automatically adding content roots to `PYTHONPATH` to run that script.
