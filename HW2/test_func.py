import sys
import torch
import json
from compute import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

def evaluate_model():
    """
    Evaluate the performance of a trained model on test data and calculate BLEU score.

    Args:
        None

    Returns:
        None
    """

    # Load the trained model
    model = torch.load('SavedModel/model0.h5', map_location=lambda storage, loc: storage)

    dataset = test_data('{}'.format(sys.argv[1]))

    # Create a data loader for testing
    testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    # Load the word-to-index mapping
    with open('i2w.pickle', 'rb') as handle:
        i2w = pickle.load(handle)

    # Move the model to GPU
    model = model.cuda()

    # Generate captions for test data using the trained model
    ss = test(testing_loader, model, i2w)

    # Save the generated captions to a file
    with open(sys.argv[2], 'w') as f:
        for id, s in ss:
            f.write('{},{}\n'.format(id, s))

    # Load the ground truth captions for test data
    test = json.load(open('/scratch1/darumil/DL/testing_label.json'))

    # Load the generated captions from the file
    output = sys.argv[2]
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption

    # Calculate BLEU score for each test item and average it
    bleu_scores = []
    for item in test:
        captions = [x.rstrip('.') for x in item['caption']]
        bleu_scores.append(BLEU(result[item['id']], captions, True))
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print("Average BLEU score is " + str(average_bleu_score))

# Call the evaluate_model function
evaluate_model()
