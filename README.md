# Personalized response generation

## Quickstart

### Access Data
Access data [here](https://drive.google.com/drive/u/0/folders/1QUlBhZmDHFXlbyOHA_ID5_kW8dej1x9I). To train a model, you can simply download all three `interview_qa_*.tsv`'s to your `data` folder.

Also, there is a `Dataset` class implemented in `interview_dataset.py` (in the project root directory) that can be used to access the data in these `tsv` files. Check that file for sample usage.

See the [Data](#data) section for details.

### Train
Right now, training for BART, BART with background, BART with Wiki context, DialoGPT, Seq2Seq and Speaker model is supported. See the [Training](#training) section for details.

### Evaluate
After you train your model, you can go to the `evaluation` folder to run the evaluation script for your model. See the [Evaluation](#model-evaluation) section for details.

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

This will generate `interview.txt` under `data`. While generating `interview.txt` from raw transcripts, this script checks whether a script is in English using [langdetect](https://pypi.org/project/langdetect/) and records non-English interviews in `non_English_interviews.txt` (each sport category has this file under its `data/[sport_type]` folder). Note that non-English interviews are prevalent among baseball, golf and soccer interviews.

### `interview_qa.tsv` and `interview_qa_*.tsv`'s
These files are in `tsv` format because dialogues naturally contain commas, and thus `csv` format isn't suitable here. `interview_qa.tsv` and its three train/dev/test splits (`interview_qa_train.tsv`, `interview_qa_dev.tsv` and `interview_qa_test.tsv`) are generated from `interview.txt`. Bascially, these files are tabulated versions of `interview.txt` and thus can be used more easily during training.

Each one of these contains a header on its first line, which has 13 fields (columns): id, sport_type, game_wiki_id, game_wiki, section_wiki_id, section_wiki, title, date, participants, background, respondent, question and response. Then from the second line onward, each line will be a single training sample.

For game_wiki and section_wiki, the method for generating those is by selecting the top 3 most similar sentences (cosine similarity of tf-idf weighted vector) by comparing question and sentences segmented from all game/section wiki texts. 

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

### General training guideline
Scripts for training are located in `train`, and training result will be located in `model_weights`.

Training result consists of model weight files and logging file. The naming convention of model weight files is `[model_name]_bsz_[#]_epoch_[#].pt` and the naming convention of logging file is `[model_name]_bsz_[#].log`. Unlike logging file, which will be one single file per training, model weights will be saved after each epoch. Hence, suppose you train one model for 5 epochs, you will have 5 model weight files and one logging file in the end.

Also, after training, there will be a `.png` file saved to `figures`, which plots loss and perplexity.

### Training BART

```python
python train_BART.py -e [epoch] -b [batch size]
```

Arguments:

`--gpu`: which gpu to use, with default value 0

`-e`: number of epochs, with default value 5

`-b`: batch size, with default value 2

### Training DialoGPT

```python
python train_DialoGPT.py -e [epoch] -b [batch size]
```

Arguments:

`--gpu`: which gpu to use, with default value 0

`-e`: number of epochs, with default value 5

`-b`: batch size, with default value 1

### Training Seq2Seq/Speaker

```python
python train_seq2seq.py -e [epoch] -b [batch size]
```
Arguments:

`-s`: toggle to train speaker model instead of Seq2Seq

`--gpu`: which gpu to use, with default value 0

`-e`: number of epochs, with default value 5

`-b`: batch size, with default value 2

`--max_grad_norm`: gradient clipping, with default value 1

## Model Evaluation

### General evaluation guideline
Scripts for model evaluation are located in `evaluation`, and evaluation result will be located in `evaluation_results`.

Each distinct model has its own evaluation script. Running the evaluation script for a model will compute perplexity, BLEU and BERT scores on the test dataset. The evaluation result will be logged into a file with `.ev` extension under `model_weights`. If you run the script multiple times, previous results won't be erased, but concatenated to that file instead. This is helpful for comparing results of different decoding schemes.

All evaluation scripts have the following arguments:

`--gpu`: which gpu to use, with default value 0

`--eval_batch_size`: batch size used for evaluation

`-e`: number of epochs trained for the model

`-b`: batch size of the model

For `-e` and `-b`, you can first go to `model_weights` to check the batch size and epoch number of the model before running the evaluation script for that model.

### Evaluating BART

To evaluate this model, inside `evaluation`, run:

```python
python eval_BART.py ...
```

There are many arguments for this script. Without any argument, this sciprt will use greedy decoding (equivalent to setting `--num_beams 1`).

For example, to use sampling based decoding with top-p probability 0.93, run:

```python
python eval_BART.py -s --top_p 0.93
```

As an another example, to use beam search with size 10 and GPU 3, run:

```python
python eval_BART.py --gpu 3 --num_beams 10
```

Script specific arguments (in addition to the general arguments described [above](#general-evaluation-guideline)):

`--num_beams`: beam search size, with default value 1

`-s`: toggle to use sampling based methods instead of beam search

`--temperature`: temperature, with default value 1.0

`--top_k`: default value 50

`--top_p`: default value 1.0

### Evaluating Seq2Seq

To evaluate this model, inside `evaluation`, run:

```python
python eval_seq2seq.py ...
```

Currently we only support greedy decoding for this model (this does not affect perplexity though). See the general arguments described [above](#general-evaluation-guideline).

### Evaluating Speaker model

To evaluate this model, inside `evaluation`, run:

```python
python eval_speaker.py ...
```

Currently we only support greedy decoding for this model (this does not affect perplexity though). See the general arguments described [above](#general-evaluation-guideline).

## Scraping

### Scraping interview transcripts
All scraped interview transcripts are located in `data/[sport_name]` folder. You don't need to redo this step since all interviews have been scrpaed.

However, if you want to scrape interviews by yourself, inside `scraping` folder, run:

```python
python scraper.py
```

The scraper will also create an `excluded_url.txt` file in each `data/[sport_name]` folder. This file documents all urls that are excluded while scraping because their contents are unable to decode properly for any reason.

### Scraping Wikipedia pages

TO BE FINISHED
