import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))

spacy.cli.download('en_core_web_lg')
nlp = spacy.load('en_core_web_lg')


def pc_score(context, response):
    context_nes = set(c.text.lower() for c in nlp(context).noun_chunks
                      if c.text.lower() not in stopwords.words('english'))
    response_nes = set(c.text.lower() for c in nlp(response).noun_chunks
                       if c.text.lower() not in stopwords.words('english'))

    count = 0
    for res_ne in response_nes:
        if res_ne in context_nes:
            count += 1

    return count / len(context_nes)


def main():
    context = '''
    And in context of the race, Henrik is proving a fair old frontrunner, isn’t he. You saw him up close today.
    '''
    response = '''
    Yeah, he’s a great player. I think he’s got a lot of game, and it’s going to be tough to catch him. But I think
    you’ve just got to go out there and play your own game and try to make as many birdies as you can and see what
    happens.
    '''
    print(pc_score(context, response))


if __name__ == '__main__':
    main()
