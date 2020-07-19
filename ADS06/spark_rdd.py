from datetime import datetime as dt
import re


def get_hour(rec):
    time = dt.utcfromtimestamp(rec['created_at_i'])
    return time.hour


def get_words(line):
    return re.compile(r'\w+').findall(line)


def count_elements_in_dataset(dataset):
    """
    Given a dataset loaded on Spark, return the
    number of elements.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: number of elements in the RDD
    """
    return dataset.count()


def get_first_element(dataset):
    """
    Given a dataset loaded on Spark, return the
    first element
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: the first element of the RDD
    """
    return dataset.first()


def get_all_attributes(dataset):
    """
    Each element is a dictionary of attributes and their values for a post.
    Can you find the set of all attributes used throughout the RDD?
    The function dictionary.keys() gives you the list of attributes of a dictionary.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: all unique attributes collected in a list
    """
    all_attributes = dataset.flatMap(lambda x: x.keys())
    unique_attributes = all_attributes.distinct()
    return unique_attributes.collect()


def get_elements_w_same_attributes(dataset):
    """
    We see that there are more attributes than just the one used in the first element.
    This function should return all elements that have the same attributes
    as the first element.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD containing only elements with same attributes as the
    first element
    """
    first_attributes = list(dataset.first().keys())
    return dataset.filter(lambda x: list(x.keys()) == first_attributes)


def get_min_max_timestamps(dataset):
    """
    Find the minimum and maximum timestamp in the dataset
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: min and max timestamp in a tuple object
    :rtype: tuple
    """
    def extract_time(timestamp):
        return dt.utcfromtimestamp(timestamp)

    timestamps = dataset.map(lambda x: x['created_at_i']).collect()
    min_time = extract_time(min(timestamps))
    max_time = extract_time(max(timestamps))
    return (min_time, max_time)


def get_number_of_posts_per_bucket(dataset, min_time, max_time):
    """
    Using the `get_bucket` function defined in the notebook (redefine it in this file), this function should return a
    new RDD that contains the number of elements that fall within each bucket.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :param min_time: Minimum time to consider for buckets (datetime format)
    :param max_time: Maximum time to consider for buckets (datetime format)
    :return: an RDD with number of elements per bucket
    """
    def get_bucket(rec, min_timestamp, max_timestamp):
        interval = (dt.timestamp(max_timestamp) - dt.timestamp(min_timestamp) + 1) / 200.0
        return int((rec['created_at_i'] - dt.timestamp(min_timestamp))/interval)

    buckets_count = dataset.map(lambda rec: (get_bucket(rec, min_time, max_time), 1))
    return buckets_count.reduceByKey(lambda buc1, buc2: buc1+buc2)


def get_number_of_posts_per_hour(dataset):
    """
    Using the `get_hour` function defined in the notebook (redefine it in this file), this function should return a
    new RDD that contains the number of elements per hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with number of elements per hour
    """
    hours_rdd = dataset.map(lambda rec: (get_hour(rec), 1))
    return hours_rdd.reduceByKey(lambda h1, h2: h1+h2)


def get_score_per_hour(dataset):
    """
    The number of points scored by a post is under the attribute `points`.
    Use it to compute the average score received by submissions for each hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with average score per hour
    """
    point_rdd = dataset.map(lambda rec: (get_hour(rec), (rec['points'], 1)))
    point_hours_rdd = point_rdd.reduceByKey(lambda left, right: (left[0] + right[0], left[1] + right[1]))
    return point_hours_rdd.map(lambda pair: (pair[0], pair[1][0] / pair[1][1]))


def get_proportion_of_scores(dataset):
    """
    It may be more useful to look at sucessful posts that get over 200 points.
    Find the proportion of posts that get above 200 points per hour.
    This will be the number of posts with points > 200 divided by the total number of posts at this hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of scores over 200 per hour
    """
    rdd1 = dataset.map(lambda rec: (get_hour(rec), (1, 1)) if rec['points'] > 200 else (get_hour(rec), (0, 1)))
    rdd2 = rdd1.reduceByKey(lambda left, right: (left[0] + right[0], left[1] + right[1]))
    return rdd2.map(lambda rec: (rec[0], rec[1][0]/rec[1][1]))


def get_proportion_of_success(dataset):
    """
    Using the `get_words` function defined in the notebook to count the
    number of words in the title of each post, look at the proportion
    of successful posts for each title length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of successful post per title length
    """
    rdd1 = dataset.map(lambda rec: (len(get_words(rec['title'])),
                                    (rec['points'], 1)) if 'title' in rec.keys() else (0, (rec['points'], 1)))
    rdd2 = rdd1.map(lambda rec: (rec[0], (rec[1][0], 1)) if rec[1][0] > 200 else (rec[0], (0, 1)))
    rdd3 = rdd2.reduceByKey(lambda left, right: (left[0] + right[0], left[1] + right[1]))
    return rdd3.map(lambda rec: (rec[0], rec[1][0]/rec[1][1]))


def get_title_length_distribution(dataset):
    """
    Count for each title length the number of submissions with that length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the number of submissions per title length
    """
    words_title_rdd = dataset.map(lambda rec: (len(get_words(rec['title'])), 1) if 'title' in rec.keys() else (0, 1))
    return words_title_rdd.reduceByKey(lambda w1, w2: w1 + w2)
