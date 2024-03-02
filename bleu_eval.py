import math
import operator
import sys
import json
from functools import reduce 

def count_ngram(candidate_sentences, reference_sentences, n):
    """
    Count the n-grams in the candidate sentences and their corresponding reference sentences.

    Args:
        candidate_sentences (list): List of candidate sentences.
        reference_sentences (list): List of reference sentences.
        n (int): The length of the n-gram.

    Returns:
        tuple: A tuple containing the precision and brevity penalty values.

    """
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate_sentences)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in reference_sentences:
            ref_sentence = reference[si]
            ngram_dict = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentence considering the n-gram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_dict.keys():
                    ngram_dict[ngram] += 1
                else:
                    ngram_dict[ngram] = 1
            ref_counts.append(ngram_dict)
        # candidate
        cand_sentence = candidate_sentences[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += calculate_clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        precision = 0
    else:
        precision = float(clipped_count) / count
    brevity_penalty_value = brevity_penalty(c, r)
    return precision, brevity_penalty_value


def calculate_clip_count(candidate_dict, reference_dicts):
    """Count the clip count for each ngram considering all references
    
    Args:
        candidate_dict (dict): A dictionary containing the ngrams and their counts in the candidate sentence.
        reference_dicts (list): A list of dictionaries, where each dictionary contains the ngrams and their counts in a reference sentence.
        
    Returns:
        int: The total clip count for all ngrams considering all references.
    """
    count = 0
    for m in candidate_dict.keys():
        m_w = candidate_dict[m]
        m_max = 0
        for ref in reference_dicts:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """
    Find the closest length of reference to that of candidate.

    Args:
        ref_l (list): A list of reference lengths.
        cand_l (int): The length of the candidate.

    Returns:
        int: The reference length that is closest to the candidate length.
    """
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    """
    Calculate the brevity penalty for a given candidate length (c) and reference length (r).

    Parameters:
    c (int): The length of the candidate sentence.
    r (int): The length of the reference sentence.

    Returns:
    float: The brevity penalty value.

    """
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    """
    Calculates the geometric mean of a list of precisions.

    Parameters:
    precisions (list): A list of precision values.

    Returns:
    float: The geometric mean of the precisions.
    """
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s, t, flag=False):
    """
    Calculate the BLEU score between a candidate sentence and reference sentence(s).

    Args:
        s (str): The candidate sentence.
        t (str or list): The reference sentence(s). If `flag` is True, `t` should be a list of reference sentences.
        flag (bool, optional): Indicates whether multiple reference sentences are provided. Defaults to False.

    Returns:
        float: The BLEU score.

    """
    score = 0.  
    count = 0
    candidate = [s.strip()]
    if flag:
        references = [[t[i].strip()] for i in range(len(t))]
    else:
        references = [[t.strip()]] 
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score


if __name__ == "__main__" :
    test = json.load(open('testing_label.json','r'))
    output = sys.argv[1]
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))