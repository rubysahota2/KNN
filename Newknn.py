import pandas as pd
import math
import operator
from sklearn import preprocessing
import random

#importing csv by the pandas using read_csv method
def load_csv(filename):
    dataset= pd.read_csv(filename)
    return dataset

#getting the number of nearest neighbours to be used
def get_k(columns):
    y = math.sqrt(columns)
    return math.floor(y)

#making the categorical data numeric and then scaling the data
def data_preprocessing(dataset):
    make_continous = ['Is declined', 'isForeignTransaction', 'isHighRiskCountry', ]
    for column in make_continous:
        dataset[column] = dataset[column].replace({"Y": 1, "N": 0})

    sub_dataset = preprocessing.scale(dataset.iloc[:, : 10])
    sub_dataset = pd.DataFrame({'Merchant_id': sub_dataset[:, 0], 'Average Amount/transaction/day': sub_dataset[:, 1], \
                                'Transaction_amount': sub_dataset[:, 2], 'Is declined': sub_dataset[:, 3], \
                                'Total Number of declines/day': sub_dataset[:, 4],
                                'isForeignTransaction': sub_dataset[:, 5], \
                                'isHighRiskCountry': sub_dataset[:, 6], 'Daily_chargeback_avg_amt': sub_dataset[:, 7], \
                                '6_month_avg_chbk_amt': sub_dataset[:, 8], '6-month_chbk_freq': sub_dataset[:, 9]})

    sub_dataset['isFradulent'] = dataset['isFradulent']
    dataset = sub_dataset.values.tolist()
    return dataset

#splitting the dataset in ratio of 0.66 testing and 0.33 in testing
def split_dataset(dataset,split_ratio):
    for x in range(len(dataset) - 1):
        for y in range(10):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split_ratio:
            training_set.append(dataset[x])
        else:
            testing_set.append(dataset[x])

#calculating the euclidian distance between the test instance and training data points
def euclideanDistance(val1, val2, length):
	distance = 0
	for x in range(length):
		distance += pow((val1[x] - val2[x]), 2)
	return math.sqrt(distance)

#getting the k-nearest neighbours , 3 nearest neighbours in this case
def getNeighbors(training_set, instance, k):
	distances = []
	length = len(instance)-1

	for i in range(len(training_set)):
		dist = euclideanDistance(instance, training_set[i], length)
		distances.append((training_set[i], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

#making a dictionary of classvotes and calculating which class got max votes, returning the class with max votes
def getResponse(neighbors):
    classVotes = {}
    for j in range(len(neighbors)):
        response = neighbors[j][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

#calculating various metrics like confusion matrix,accuracy,prediction and fscore
def confusion_matrix(testing_set,predictions):
    tp=0
    fp=0
    tn=0
    fn=0

    for j in range(len(testing_set)):

        if((testing_set)[j][-1]=='Y' and predictions[j]=='Y'):
            tp += 1
        if((testing_set)[j][-1]=='N' and (predictions[j]=='Y')):
            fp += 1
        if ((testing_set)[j][-1] == 'N' and (predictions[j] == 'N')):
            tn += 1
        if ((testing_set)[j][-1] == 'Y' and (predictions[j] == 'N')):
            fn += 1

    matrixx = [['*  ','Pred:Y','Pred:N'],['Act:Y',tp,fn],['Act:N',fp,tn]]
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in matrixx]))

    precison = tp / (tp + fp) * 100
    f_score=(2*tp)/((2*tp)+fp+fn)
    accuracy= (tp+tn)/len(testing_set)*100
    print('Accuracy : ' + repr(accuracy)+'%')
    print('Precision: ' + repr(precison) + '%')
    print('F Score: ' + repr(f_score))


filename = 'creditcardcsvpresent.csv'
dataset = load_csv(filename)
columns= len(dataset.columns)
dataset=data_preprocessing(dataset)
training_set=[]
testing_set=[]
split_dataset(dataset,split_ratio=0.66)
predictions=[]
k= get_k(columns)

for j in range(len(testing_set)):
    instance= testing_set[j]
    neighbors = getNeighbors(training_set, instance, k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testing_set[j][-1]))
confusion_matrix(testing_set,predictions)




