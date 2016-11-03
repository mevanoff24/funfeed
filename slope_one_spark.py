
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.sql import SQLContext
import operator
import math


conf = SparkConf().setAppName("Slope One")
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'
sqlContext = SQLContext(sc)




def parse_lines(line):
    fields = line.split(',')
    item = int(fields[0])
    user_id = int(fields[1])
    rating = int(fields[2])
    return (user_id, item, rating)


def main():

    lines = sc.textFile('data/sample_train_int.csv')
    header = lines.first() #extract header
    lines = lines.filter(lambda x: x != header)
    # training_in = lines.map(get_tuple)

    training_data = lines.map(parse_lines)
    # training_data = training_data.sample(False, 100000)

    training_RDD, val_RDD = training_data.randomSplit([8, 2], seed=0L)
    print('Training Data Points', training_RDD.count())
    print('Validation Data Points', val_RDD.count())



    training_df = sqlContext.createDataFrame(training_RDD, ['uid', 'iid', 'rating'])
    testing_df = sqlContext.createDataFrame(val_RDD, ['uid', 'iid', 'rating'])

    training_df.registerTempTable("TrainingTable")
    testing_df.registerTempTable("TestingTable")

    # calculate the deviation between each item pairs. 
    # dev(j,i) = sum(r_j-r_i)/c(j,i)

    # all difference between ratings
    joined_user_df = sqlContext.sql("""
    SELECT t1.uid, t1.iid as iid1, t2.iid as iid2, (t1.rating - t2.rating) as rating_diff FROM
    TrainingTable t1
    JOIN
    TrainingTable t2
    ON (t1.uid = t2.uid)
    """)
    # joined_user_df.show()

    # sum / count of rating difference 
    # |iid1|iid2|                 dev|  c|
    joined_user_df.registerTempTable("JoinedUserTable")
    mpair_dev_c_df = sqlContext.sql("""
    SELECT iid1, iid2, sum(rating_diff) / count(rating_diff) as dev, count(rating_diff) as c FROM
    JoinedUserTable
    Group By iid1, iid2
    """)
    # mpair_dev_c_df.show()

    testing_training_df = sqlContext.sql("""
    SELECT t1.uid, t1.iid as iidj, t2.iid as iidi, t1.rating as rating_j, t2.rating as rating_i FROM
    TestingTable t1
    JOIN
    TrainingTable t2
    ON (t1.uid = t2.uid)
    """)
    # testing_training_df.show()

    # # join tables
    join_cond = [testing_training_df.iidj == mpair_dev_c_df.iid1, testing_training_df.iidi == mpair_dev_c_df.iid2]
    df = testing_training_df.join(mpair_dev_c_df, join_cond)
    # df.show()

    
    # calculate how likely a user in the testing set will like an item. 
    # P(a,j) = sum((dev(j,i)+r(a,i))*c(j,i))/sum(c(j,i))
    df.registerTempTable("AllTable")
    ps = sqlContext.sql("""
    SELECT uid, iidj, sum((dev + rating_i) * c) / sum(c) as p, rating_j as true_rating FROM
    AllTable
    Group By uid, iidj, rating_j
    """)
    # ps.show()

    # # calculate RMSE
    ps.registerTempTable("PTable")
    rmse = sqlContext.sql("""
    SELECT sqrt(sum(power(true_rating - p, 2))/count(true_rating)) as RMSE FROM
    PTable
    """)
    rmse.show()

if __name__ == '__main__':
    main()





# 100000
# |1.7743629756741233|
# 200000
#|1.805043964351539|
# 300000
# |1.808744114269386|
# 400000
# |1.7948057653789624|