import os
from json import JSONEncoder
import nltk
import numpy as np
from flask import Flask, render_template, request, redirect, \
    url_for, flash, make_response, session
from flask import request
from datetime import datetime
from app.search_engine.algorithms import build_terms, create_index_tfidf
from app.search_engine.load_corpus import load_corpus, corpus_to_dict
from app.analytics.analytics_data import AnalyticsData, ClickedDoc, Term, User
from app.search_engine.objects import Document, StatsDocument
from app.search_engine.search_engine import SearchEngine
import httpagentparser
import time

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b6st8b76va8er76fcs6g8d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# READ DATA
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + '/app/search_engine/dataset_tweets_WHO.txt'

#LOAD CORPUS
corpus, corpus_dict = load_corpus(file_path)

#CREATE INDEX
print("Creating index, please wait...")
length = len(corpus_dict)
for key in range(length):
    corpus_dict[key]['full_text'] = build_terms(corpus_dict[key]['full_text'])
start_time = time.time()
num_tweets = len(corpus_dict)
index, tf, df, idf, title_index = create_index_tfidf(corpus_dict, num_tweets)
print("Total time to create the index: {} seconds".format(np.round(time.time() - start_time, 2)))
print("loaded corpus. first elem:", list(corpus.values())[0])

# instances
search_engine = SearchEngine(idf, tf, title_index, index)
analytics_data = AnalyticsData()

@app.route('/')
def search_form():

    #get user info
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    analytics_data.fact_users[user_ip] = user_agent

    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']

    #stats
    for word in search_query.split():
        if word in analytics_data.fact_terms.keys():
            doc_counter = analytics_data.fact_terms[word]
            analytics_data.fact_terms[word] = doc_counter + 1
            analytics_data.fact_nterms[word] = sum([analytics_data.fact_nterms[word], len(search_query.split())]) / 2
        else:
            analytics_data.fact_terms[word] = 1
            analytics_data.fact_nterms[word] = 1

        analytics_data.fact_date[word] = datetime.now()

    #results
    results = search_engine.search(search_query, corpus_dict)
    docs= []
    for result in results:
        d: Document = corpus[result.id]
        d.url=result.url
        docs.append(d)
    found_count = len(results)

    return render_template('results.html', results_list=docs, page_title="Results", found_counter=found_count)


@app.route('/doc_details', methods=['GET', 'POST'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')
    clicked_doc_id = int(request.args.get('id'))
    doc = corpus[int(clicked_doc_id)]

    if clicked_doc_id in analytics_data.fact_clicks.keys():
        doc_counter = analytics_data.fact_clicks[clicked_doc_id]
        analytics_data.fact_clicks[clicked_doc_id] = doc_counter + 1

    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    return render_template('doc_details.html', doc=doc)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """

    #doc clicks
    docs = []
    for doc_id in analytics_data.fact_clicks:
        print(doc_id)
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)
    docs.sort(key=lambda doc: doc.count, reverse=True) #simulate sort by ranking

    #terms
    terms = []
    for word in analytics_data.fact_terms:
        term_counts = Term(word, analytics_data.fact_terms[word], analytics_data.fact_date[word], analytics_data.fact_nterms[word])
        terms.append(term_counts)
    terms.sort(key=lambda terms: terms.count, reverse=True)

    users = []
    for user_ip in analytics_data.fact_users:
        user = User(user_ip, analytics_data.fact_users[user_ip])
        users.append(user)

    return render_template('stats.html', clicks_data=docs, terms=terms, users=users)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    visited_docs = []
    print(analytics_data.fact_clicks.keys())
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    terms = []
    for word in analytics_data.fact_terms:
        term_counts = Term(word, analytics_data.fact_terms[word], analytics_data.fact_date[word], analytics_data.fact_nterms[word])
        terms.append(term_counts)
    terms.sort(key=lambda terms: terms.count, reverse=True)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    return render_template('dashboard.html', visited_docs=visited_docs, terms=terms)

if __name__ == "__main__":
    app.run(port="8088", host="0.0.0.0", threaded=False, debug=False)