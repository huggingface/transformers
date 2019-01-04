import os
import json
import numpy as np
from konlpy.tag import Okt


class ContextFinder:

    # BASE_PATH : Project path
    # vectorize : Initialize TF-IDF matrix
    # documents : documents ( contexts in selected title or paragraph )
    # X : Generated TF-IDF weights matrix after fitting input documents
    # features : a.k.a vocabulary
    # tokenizer : Open-Korean-Text for Korean language processing
    def __init__(self):
        self.BASE_PATH = os.path.dirname(os.path.abspath(__name__))
        self.vectorize = self.init_tf_idf_vector()
        self.documents = []
        self.X = None
        self.features = None
        self.tokenizer = Okt()

    # tokenization
    # norm : ㅋㅋㅋㅋㅋ ---> ㅋㅋ
    # stem : 들어간다 ---> 들어가다.
    def convert_to_lemma(self, text: str) -> list:
        return self.tokenizer.morphs(text, norm=True, stem=True)

    # for testing, pos tagged tuple list
    def check_pos(self, text: str) -> list:
        return self.tokenizer.pos(text)


    # loading dev data
    def load_context_by_title(self, dataset_path):
        if dataset_path is None:
            dataset_path = 'dev-v1.1.json'
        with open(dataset_path) as f:
            data = json.load(f)['data']
            for article in data:
                for paragraph in article.get('paragraphs'):
                    self.documents.append(paragraph.get('context'))

    # initializing vectorizer object, adding custom tokenizer above (convert_to_lemma)
    def init_tf_idf_vector(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            tokenizer=self.convert_to_lemma,
            min_df=1,
            sublinear_tf=True
        )

    def generate_tf_idf_vector(self):
        self.X = self.vectorize.fit_transform(self.documents)
        self.features = self.vectorize.get_feature_names()
        # e.g) after fitting 5 sentences and 7 features, matrix X looks like below
        # ([[0.        , 0.40824829, 0.81649658, 0.        , 0.        , 0.        , 0.40824829],
        # [0.        , 0.40824829, 0.40824829, 0.        , 0.        , 0.        , 0.81649658],
        # [0.41680418, 0.        , 0.        , 0.69197025, 0.41680418, 0.41680418, 0.        ],
        # [0.76944707, 0.        , 0.        , 0.63871058, 0.        , 0.        , 0.        ],
        # [0.        , 0.        , 0.        , 0.8695635 , 0.34918428, 0.34918428, 0.        ]])

    def build_model(self, dataset_path=None):
        self.load_context_by_title(dataset_path)
        self.generate_tf_idf_vector()

    def get_ntop_context(self, query: str, n: int) -> str:
        if self.X is None or self.features is None:
            self.build_model()

        # check input query keywords if they are in feature(vocabulary)
        keywords = [word for word in ContextFinder.convert_to_lemma(query) if word in self.features]

        # get indexes of keywords in X( TF-IDF matrix )
        matched_keywords = np.asarray(self.X.toarray())[:, [self.vectorize.vocabulary_.get(i) for i in keywords]]
        #       word 1      word 2
        # 0     0.000000    0.000000 doc 1
        # 1     0.000000    0.000000 doc 2
        # 2     0.416804    0.691970 doc 3
        # 3     0.769447    0.638711 doc 4
        # 4     0.000000    0.869563 doc 5

        # sum each words weights document by document and sorting reverse order
        scores = matched_keywords.sum(axis=1).argsort()[::-1]
        for i in scores[:n]:
            if scores[i] > 0:
                yield self.documents[i]

    def get_ntop_context_by_cosine_similarity(self, query: str, n: int):
        from sklearn.metrics.pairwise import linear_kernel
        if self.X is None or self.features is None:
            self.build_model()
        query_vector = self.vectorize.transform([query])

        # linear_kernel is dot product between query_vector and all documents vector and transform 1 dim array
        cosine_similar = linear_kernel(query_vector, self.X).flatten()
        ranked_idx = cosine_similar.argsort()[::-1]
        for i in ranked_idx[:n]:
            if cosine_similar[i] > 0:
                yield self.documents[i]


## usages
if __name__ == "__main__":

    # dev set articles
    c = ContextFinder()
    print(c.convert_to_lemma("미국 군대 내 두번째로 높은 직위는 무엇인가?"))
    # c.build_model('./bert_chatbot/dev-v1.1.json')
    # for i in c.get_ntop_context('what is doctor who?', 5):
    #     print(i)
    #
    # for i in c.get_ntop_context_by_cosine_similarity('what is doctor who?', 5):
    #     print(i)
