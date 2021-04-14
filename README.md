# Personalized response generation

## Quickstart

### Data
Access data [here](https://drive.google.com/drive/u/0/folders/1QUlBhZmDHFXlbyOHA_ID5_kW8dej1x9I). For training models, you can simply download all three `interview_qa_*.tsv`'s to your `data` folder. These files are in `tsv` format because dialogues naturally contain commas, and thus `csv` format isn't suitable here.

Also, there is a `Dataset` class implemented in `interview_dataset.py` (in the project root directory) that can be used to access the data in these `tsv` files. Check that file for sample usage.

See the [Data](#data) section for details.

### Train
To be done...

## Data

### `interview.txt`
This file contains structured interviews generated directly from scraped interview transcripts. Also, Wikipedia pages associated for each interview is also documented in this file.
Each interivew transcipt is separated by a special token `[SEP]`. For each interview, the structure of the data looks similar to this:

1 [id] actual_id

2 [sport_type] actual_sport_type

3 [game_wiki] game_wiki_id

4 [section_wiki] section_wiki_id

5 [title] actual_title

6 [date] actual_date

7 [participants] actual_participant_1|actual_participant_2|actual_participant_3

8 [background] some_background

9 [background] some_background

10 [QA] Q: actual_question \t INTERVIEWEE_NAME: actual_answer

11 [QA] Q: actual_question \t INTERVIEWEE_NAME_1: actual_answer_1 \t INTERVIEWEE_NAME_2: actual_answer_2

[SEP]


Notes: 1-7 are fixed, while [background] and [QA] sections are of variable lengths. Also, there maybe no [game_wiki] or [section_wiki] for some interviews. In this case, the [game_wiki] or the [section_wiki] tag will still be present (with an empty string following it indicating such information is not applicable). There may also be cases where no [background] is aviable. In this case, there will be no line with a [background] tag ([QA] section will directly follow the [participants] section).

If you want to generate `interview.txt` by yourself, inside `processing` folder, run:

```python
python generate_dataset.py
```

This will generate `interview.txt` under `data`. While generating `interview.txt` from raw transcripts, this script checks whether a script is in English by using [langdetect](https://pypi.org/project/langdetect/) and writes down interviews that are not English into a file `non_English_interviews.txt` for each sport category under its respective folder `data/[sport_type]`. Note that non-English interviews are prevalent among baseball, golf and soccer interviews.

### `interview_qa.tsv` and `interview_qa_*.tsv`'s
`interview_qa.tsv` and its three train/dev/test splits (`interview_qa_train.tsv`, `interview_qa_dev.tsv` and `interview_qa_test.tsv`) are generated from `interview.txt`. Bascially, these files are tabulated versions of `interview.txt` and thus can be used more easily during training.

Each one of these contains a header on its first line, which has 11 fields (columns): id, sport_type, game_wiki, section_wiki, title, date, participants, background, respondent, question and response. Then from the second line onward, each line will be a single training sample.

The train/dev/test splitting is partioned using a ratio of 0.98:0.1:0.1. The splitting is performed after the original dataset is shuffled randomly.

If you want to generate there files by yourself, inside `processing` folder, run:

```python
python generate_tsv.py
```

This will generate all these files under `data`.

### Pytorch `Dataset` class
There is a class `InterviewDataset` that extends `torch.utils.data.Dataset` for this custom dataset. This class is inside `interview_dataset.py` (in the project root directory). You can use this class along with a standard Pytorch `DataLoader`. A sample use case is also included in that script.

While instantiating the class, there is an optional parameter `use_wiki` which defaults to `False`. If you want to use Wikipedia pages for your model, set this parameter to `True`.

The `__getitem__` function returns a dictionary of length 11. The keys of this dictionary are the 11 fields described above. Note that if any of the fileds is empty for a training sample, then the value of that field will be just an empty string.

### LEGACY Build vocabulary
No need to perform this step again because the vocabulary files are already generated and pushed to Github, under `data/vocab`. The current vocabulary size is 161,292.

If there are new interview texts available and you need to rebuild the vocabulary, first re-generate all csv files by following the instruction above, then inside `processing` folder, run:

```python
python build_vocab.py
```

This will generate the vocabulary files again, and remove the older vocabulary files.

### LEGACY Link game background with each episode of interview
To link wikipedia pages for each episode of interview, inside `processing`, run:

```python 
python link_game_background.py
```

This script requires `game_search_result` and `section_search_result` which are loacted in `data/`. This script will scrape and assign an index for each wikipedia page and then it appends two columns at the end of `*_episode.csv` indicating the index of the background wikipedia pages.

## Training
### Training BART-base or seq2seq
Get smaller version of training data [here](https://drive.google.com/drive/folders/1QUlBhZmDHFXlbyOHA_ID5_kW8dej1x9I?usp=sharing). To use the data, download all `smaller_utterance_*.csv` to `data/csv`.

For example, to train BART-base, inside `training`, run:

```python
python train_BART.py -e [epoch] -b [batch size]
```

Arguments of these scripts:

`--gpu`: which gpu to use, with default value 0

`-e`: number of epochs, with default value 5

`-b`: batch size, with default value 2

### Evaluating trained BART-base
Get the trained model [here](https://drive.google.com/drive/folders/12wZvtyhnTpjQEqjKljWio8bQh4syt6io?usp=sharing). To use this model, download `bart-base_epoch_10_bsz_2_small_utterance.pt` to `model`.

To evaluate this particular trained model, inside `training`, run:

```python
python eval_BART.py ...
```

Running this script will compute perplexity and BLEU scores on the validation dataset. The evaluation result will be logged into a file with `.ev` extension inside `model`. If you run this script multiple times, previous results won't be erased, but concatenated to that file instead. This is helpful for comparing results of different decoding schemes.

There are many arguments for this script. Without any argument, this sciprt will use greedy decoding (equivalent to setting `--num_beams 1`).

To use sampling based decoding with top-p probability 0.93, run:

```python -W ignore eval_BART.py -s --top_p 0.93 ```

To use beam search with size 10 and GPU 3, run:

```python -W ignore eval_BART.py --gpu 3 --num_beams 10 ```

Arguments of `eval_BART.py`:

`--gpu`: which gpu to use, with default value 0

`--batch_size`: batch size used while performing evaluation, with default value 5

`--num_beams`: beam search size, with default value 1

`--temperature`: a hyperparameter for beam search, with default value 1.0

`-s`: toggle to use sampling based methods instead of beam search

`--top_k`: default value 50

`--top_p`: default value 1.0

## Scraping
### Web scraping
All scraped interview transcripts are located in `data/[sport_name]` folder. You don't need to redo this step since all interviews have been scrpaed.

However, if you want to scrape interviews by yourself, inside `scraping` folder, run:

```python
python scraper.py
```

The scraper will also create an `excluded_url.txt` file in each `data/[sport_name]` folder. This file documents all urls that are excluded while scraping because their contents are unable to decode properly for any reason.
