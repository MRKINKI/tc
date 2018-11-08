# -*- coding: utf-8 -*-

from main.baseline.data_process import load_data_from_csv, seg_words
from main.baseline.model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from main.baseline import config
import logging
import numpy as np
from sklearn.externals import joblib
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

    # load train data
    logger.info("start load data")
    train_data_df = load_data_from_csv(config.train_data_path)
    validate_data_df = load_data_from_csv(config.validate_data_path)

    content_train = train_data_df.iloc[:, 1]

    logger.info("start seg train data")
    content_train = seg_words(content_train)
    logger.info("complete seg train data")

    columns = train_data_df.columns.values.tolist()
    logger.info("start seg validate data")
    content_validate = validate_data_df.iloc[:, 1]
    content_validate = seg_words(content_validate)
    logger.info("complete seg validate data")

    logger.info("start train feature extraction")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5, norm='l2')
    vectorizer_tfidf.fit(content_train)
    
    
    train_x = vectorizer_tfidf.transform(content_train)
    print(train_x.shape)
    
    
    val_x = vectorizer_tfidf.transform(content_validate)
    print(val_x.shape)
    


    # model train
    logger.info("start train model")
    classifier_dict = dict()
    for column in columns[2:]:
        label_train = train_data_df[column]
        text_classifier = TextClassifier()
        logger.info("start train %s model" % column)
        # print(content_train.shape)
        text_classifier.fit(train_x, label_train)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = text_classifier

    logger.info("complete train model")

    # validate model
    
    logger.info("start validate model")
    f1_score_dict = dict()
    for column in columns[2:]:
        label_validate = validate_data_df[column]
        text_classifier = classifier_dict[column]
        f1_score = text_classifier.get_f1_score(val_x, label_validate)
        f1_score_dict[column] = f1_score

    f1_score = np.mean(list(f1_score_dict.values()))
    str_score = "\n"
    for column in columns[2:]:
        str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % f1_score)
    logger.info("complete validate model")

    # save model
    logger.info("start save model")
    model_save_path = config.model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    joblib.dump(vectorizer_tfidf, model_save_path + 'vetorize.pkl')
    joblib.dump(classifier_dict, model_save_path + model_name)
    logger.info("complete save model")
