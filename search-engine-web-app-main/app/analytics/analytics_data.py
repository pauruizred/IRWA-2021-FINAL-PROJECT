import json

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """

    #dicts to store stats
    fact_date = dict([])
    fact_terms = dict([])
    fact_nterms = dict([])
    fact_clicks = dict([])
    fact_users = dict([])

class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)

class Term:
    def __init__(self, term, count, last, maxlength):
        self.term = term
        self.count = count
        self.last = last
        self.maxlength = maxlength

class User:
    def __init__(self, ip, info):
        self.ip = ip
        self.info = info