import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

'''
Using SBERT to encode query and sentences from text document and using cosine similarity to calculate score
'''
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# similar to top-k sampling, choose k most similar sentences
def find_top_k(query, text, k=6):
    sentences = [s.strip() for s in text.split('.')]

    query_vec = np.expand_dims(model.encode(query), axis=0)
    sentence_vec = model.encode(sentences)

    score = cosine_similarity(query_vec, sentence_vec).squeeze()

    sorted_score = np.argsort(score)

    chosen = sorted_score[-min(len(sorted_score), k):]

    res = [sentences[i] for i in range(len(sentences)) if i in chosen]

    return res


# similar to nucleus sampling, choose sentences until cumulative probability exceeds threshold p
def find_top_p(query, text, p=0.9):
    sentences = [s.strip() for s in text.split('.')]

    query_vec = np.expand_dims(model.encode(query), axis=0)
    sentence_vec = model.encode(sentences)

    score = softmax(cosine_similarity(query_vec, sentence_vec).squeeze())

    sorted_score = np.argsort(score)

    cumulative_prob = 0
    for i in range(len(sorted_score) - 1, 0, -1):
        cumulative_prob += score[sorted_score[i]]
        if cumulative_prob >= p:
            chosen = sorted_score[i:]
            break

    res = [sentences[i] for i in range(len(sentences)) if i in chosen]

    return res


# provide an example
def main():
    q = "Did they do anything differently, James Madison? They really held you guys as far as the running game?"

    t = '''
    The 2017 NCAA Division I Football Championship Game was a postseason college football game that determined a national champion in the NCAA Division I Football Championship Subdivision for the 2016 season. It was played at Toyota Stadium in Frisco, Texas, on January 7, 2017, with kickoff at 12:00 noon EST, and was the culminating game of the 2016 FCS Playoffs. With sponsorship from Northwestern Mutual, the game was officially known as the NCAA FCS Football Championship Presented by Northwestern Mutual.
    
    Teams
    The participants of the 2017 NCAA Division I Football Championship Game were the finalists of the 2016 FCS Playoffs, which began with a 24-team bracket. No. 4 seed James Madison and unseeded Youngstown State qualified for the final by winning their semifinal games. James Madison was the designated home team for the final game.
    
    Youngstown State Penguins
    Youngstown State finished their regular season with an 8–3 record (6–2 in conference). In the FCS playoffs, they defeated Samford, Jacksonville State, Wofford, and second-seeded Eastern Washington to reach the finals. The Penguins entered the championship game with a 4–2 record in prior FCS/Division I-AA finals, contested during the 1991 through 1999 seasons.
    
    James Madison Dukes
    James Madison finished their regular season with a 10–1 record (8–0 in conference). Their only loss was to North Carolina of the FBS, 56–28. In the FCS playoffs, they defeated New Hampshire, Sam Houston State, and top-seeded North Dakota State to reach the finals. The Dukes entered the championship game with a 1–0 record in prior FCS/Division I-AA finals, having defeated Montana for the 2004 season title.
    '
    Game summary
    Scoring summary
    Game statistics
    Notes
    References
    External links
    Box score at ESPN
    January 7, 2017 - Youngstown St. vs. James Madison via YouTube
    '''

    print(find_top_p(q, t))
    print(find_top_k(q, t))


if __name__ == '__main__':
    main()
