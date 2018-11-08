#!/user/bin/env python
# -*- coding:utf-8 -*-

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        # classifier = SVC(kernel="rbf")
        # classifier = SVC(kernel="linear")
        classifier = LogisticRegression(
                                        #solver='lbfgs',
                                        C=10, 
                                       # class_weight='balanced', 
                                        n_jobs = -1)
        self.classifier = classifier

    def fit(self, x, y):

        self.classifier.fit(x, y)

    def predict(self, x):

        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)

    def get_f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='macro')



