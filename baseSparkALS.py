import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import SQLContext
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np 

conf = SparkConf().setMaster("local[*]").setAppName("ALS")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

# data = sc.textFile('data/new_train.csv')
# test = pd.read_csv('data/new_test.csv')

# le1 = LabelEncoder()
# le2 = LabelEncoder()

# item_int = le1.fit_transform(test.item)
# user_int = le2.fit_transform(test.user_id)

# test['item_int'] = item_int
# test['user_int'] = user_int

# test.drop(['item', 'user_id'], axis = 1, inplace = True)
# test = test[['user_int', 'item_int']]
# test.to_csv('data/new_test_int_ALS.csv', index = False)



# item,user_id,rating
def parse_lines(line, train=True):
	fields = line.split(',')
	if train:
		# fields = line.split(',')
		item = int(fields[0])
		user_id = int(fields[1])
		rating = int(fields[2])
		return (user_id, item, rating)
	item = int(fields[0])
	user_id = int(fields[1])
	return (user_id, item)

def parse_lines_test(line):
    fields = line.split(',')
    item = int(fields[1])
    user_id = int(fields[0])
    return (user_id, item)

def clamp_pred(line):
    np.clip(line, 1, 10)



# lines = sc.textFile('data/new_int_train.csv')
lines = sc.textFile('data/sample_train_int.csv')
header = lines.first() #extract header
lines = lines.filter(lambda x: x != header)
data = lines.map(parse_lines)



# test_lines = sc.textFile('data/new_test_int_ALS.csv')
# test_header = test_lines.first() #extract header
# test_lines = test_lines.filter(lambda x: x != test_header)
# testing_data = test_lines.map(parse_lines_test)




# Split Data
training_RDD, val_RDD = data.randomSplit([8, 2], seed=0L)
# print('Training Data Points', training_RDD.count())
# print('Validation Data Points', val_RDD.count())

train_for_predict_RDD = training_RDD.map(lambda x: (x[0], x[1]))
val_for_predict_RDD = val_RDD.map(lambda x: (x[0], x[1]))
# ratings = training_RDD.map(lambda l: Rating(l[0], l[1], float(l[2]))).cache()

# full data
# ratings = data.map(lambda l: Rating(l[0], l[1], l[2])).cache()



seed = 5L
iterations = 5
regularization_parameter = 0.1
ranks = [2]
val_errors = [0]
train_errors = [0]
err = 0

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)

    # train_predictions = model.predictAll(train_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    # train_rates_and_preds = training_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(train_predictions)
    # train_error = math.sqrt(train_rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    # train_errors[err] = train_error

    val_predictions = model.predictAll(val_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    print val_predictions.take(10)
    clamp_predictions = val_predictions.map(lambda r: np.clip(r[2], 1, 10))

    val_rates_and_preds = val_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(clamp_predictions)
    val_error = math.sqrt(val_rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    val_errors[err] = val_error

    # test_predictions = model.predictAll(testing_data).map(lambda r: ((r[0], r[1]), r[2]))

    # test_rates_and_preds = testing_data.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(test_predictions)

    # df = test_predictions.toDF()
    # df = df.toPandas()
    # print df.shape
    # df.columns = ['comb', 'pred']
    # df['user_id'] = df.comb.map(lambda x: x[0])
    # df['item'] = df.comb.map(lambda x: x[1])
    # df.drop(['comb'], axis = 1, inplace = True)
    # df = df[['user_id', 'item', 'pred']]
    # print df.head()
    # df.to_csv('data/ALS_play.csv', index = False)

    # df['item'] = le1.inverse_transform(df['item'])
    # df['user_id'] = le2.inverse_transform(df['user_id'])
    # print df.head()
    # print df.shape
    # df.to_csv('data/ALS_base.csv', index = False, header = None)

    # original_test = pd.read_csv('data/new_test_int_and_original.csv')
    # print original_test.head()
    # output = pd.merge(original_test, df, left_on = ['user_int', 'item_int'], right_on = ['user_id', 'item'])
    # print output.head()
    # print output.shape
    # output.to_csv('data/ALS_base.csv', index = False, header = None)

    # preds = []
    # user_ids = []
    # item_ids = []

    # all_pred = test_predictions.map(lambda x: ((x[2])))
    # all_user_id = test_predictions.map(lambda x: (x[0]))
    # print all_pred.take(10)
    # print all_user_id.take(10)

    # print np.array(all_pred)[:10]
    
#     err += 1

#     print('For rank %s the RMSE is %s' % (rank, val_errors))
#     if val_error < min_error:
#         min_error = val_error
#         best_rank = rank


    print '****************'
    print '****************'
    print 'The best model was trained with rank %s' % best_rank
    print train_errors
    print val_errors
    print '****************'
    print '****************'
# 1.7382155839223437


# plt.plot(train_errors, label = 'train')
# plt.plot(val_errors, label = 'validation')
# plt.legend()
# plt.show()

# print("\nTraining recommendation model...")
# rank = 10
# # Lowered numIterations to ensure it works on lower-end systems
# numIterations = 6
# model = ALS.train(ratings, rank, numIterations)




# predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
# rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
# error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
# print '****************'
# print error
# print '****************'

# userID = 100

# print("\nRatings for user ID " + str(userID) + ":")
# userRatings = ratings.filter(lambda l: l[0] == userID)
# for rating in userRatings.collect():
#     # print nameDict[int(rating[1])] + ": " + str(rating[2])
#     print rating




# print("\nTop 10 recommendations:")
# recommendations = model.recommendProducts(userID, 10)
# for recommendation in recommendations:
#     # print nameDict[int(recommendation[1])] + \
#     #     " score " + str(recommendation[2])
#     print recommendation



#### First Atempt Reg = 0.1  ranks = [4, 8, 12]
# ****************
# ****************
# The best model was trained with rank 4
# [1.2190293098609917, 1.0430165279752242, 0.9372943338016972]
# [1.7430307106885496, 1.7861848261566378, 1.8030226694946834]
# ****************
# ****************




# seed = 5L
# iterations = 5
# regularization_parameter = 0.1
# ranks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#The best model was trained with rank 2
# Train Scores
#[1.4050965007563434, 1.2739185248308944, 1.188021612800204, 1.1178986727513822, 1.0656770732095633, 1.0213323424257517, 0.9756184206125713, 0.9415197733927755, 0.9194783247327019, 0.8942681195613953]
# Validation Scores
#[1.6581690592123703, 1.727381358167416, 1.7635680973108476, 1.789139980591774, 1.8269070701543402, 1.837319132565302, 1.8324468408909675, 1.8468669617724014, 1.8656573107937962, 1.8810434660472195]




