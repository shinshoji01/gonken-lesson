# A GOOD performance classifier
score_1=[0.9,0.8,0.7,0.6,0.8,  0.85,0.66,0.7,0.95,0.58,  0.4,0.32,0.28,0.16,0.2,  0.22,0.33,0.12,0.05,0.1]
# A MEDIUM performance classifier
score_2=[0.9,0.8,0.7,0.6,0.8, 0.4,0.3,0.2,0.36,0.48,  0.4,0.32,0.28,0.16,0.2,  0.56,0.68,0.77,0.67,0.6]
# A POOR performance classifier
score_3=[0.82,0.4,0.15,0.16,0.4,  0.4,0.3,0.2,0.36,0.48,  0.21,0.8,0.77,0.63,0.82,  0.56,0.68,0.77,0.72,0.6]

truths=[True,True,True,True,True,  True,True,True,True,True,  False, False,False,False,False,  False, False,False,False,False,]

model_scores = [score_1, score_2, score_3]

#Function to draw the PR Curve
def PR(threshold_value, model_index):
    score = model_scores[model_index]
    true_positives=0
    false_positives=0
    false_negatives=0 
    # using given threshold value, calculate classification recall and precision for chosem model
    for i in range(len(truths)):
        if score[i]>=threshold_value:
            if truths[i]:
                true_positives+=1
            else:
                false_positives+=1 
        else:
            if truths[i]:
                false_negatives+=1
    if (true_positives+false_positives)>0:           
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 1
    recall = true_positives / (true_positives + false_negatives)
    return precision, recall
