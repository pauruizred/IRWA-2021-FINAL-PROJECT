from app.search_engine.algorithms import get_ranked_docs
from app.search_engine.objects import ResultItem, Document

class SearchEngine:
    """educational search engine"""

    def __init__(self, idf, tf, title_index, index):
        self.tf = tf
        self.idf = idf
        self.title_index = title_index
        self.index = index

    def search(self, search_query, tweets_data):

        ranked_docs = get_ranked_docs(search_query, tweets_data, self.index, self.idf, self.tf, self.title_index)
        
        res = []
        for i in range(len(ranked_docs)):

            id = tweets_data[ranked_docs[i]]['id']
            description = " ".join(tweets_data[ranked_docs[i]]['full_text'])
            title = description[0:30]
            date = tweets_data[ranked_docs[i]]['created_at']
            ranking = i

            res.append(ResultItem(id, title, description, date, "doc_details?id={}&title={}&date={}".format(id, title, date), ranking))
        return res

