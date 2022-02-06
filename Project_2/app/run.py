import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
#from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
try:
    engine = create_engine('sqlite:///data/DisasterResponse.db')
except:
    print("Cannot connect DB.")


df = pd.read_sql_table('disaster_response_df', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    ## -------------------------------------------
    ## 1. Dist of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ## --- make graph
    graph_one = []
    graph_one.append(
                     Bar(
                         x = genre_names,
                         y = genre_counts
                         )
                     )
    
    ## --- make layout
    layout_one = dict(
                      title = 'Distribution of Message Genres',
                      xaxis = dict(title = 'Genre'),
                      yaxis = dict(title = 'Count')
                      )
    
    ## -------------------------------------------
    ## 2. Dist of Tags
    tags_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    tags_names = list(tags_counts.index)
    
    ## --- make graph
    graph_two = []
    graph_two.append(
                     Bar(
                         x = tags_names,
                         y = tags_counts
                         )
                     )
        
    ## --- make layout
    layout_two = dict(
                      title = 'The Number of Tags; Descending',
                      xaxis = dict(title = 'Tag'),
                      yaxis = dict(title = 'Count')
                      )

    ## -------------------------------------------
    ## 3. Most Common Words in Message; Top 20
    stopwords = ['the', 'and', 'for', 'have', 'are', 'with', 'you', 'this', "n't", 'not']
    word_tmp = []
    for i in range(len(df['message'])):
        word_tmp += tokenize(df['message'][i])
    
    word_counts = pd.Series([word for word in word_tmp if (len(word)>2) and (word not in stopwords)])
    word_counts = word_counts.value_counts().sort_values(ascending=False)[:20]
    word_names = list(word_counts.index)

    ## --- make graph
    graph_three = []
    graph_three.append(
                     Bar(
                         x = word_names,
                         y = word_counts
                         )
                     )

    ## --- make layout
    layout_three = dict(
                      title = 'Most Common Words in Message; Top 20',
                      xaxis = dict(title = 'Word'),
                      yaxis = dict(title = 'Count')
                      )
    
    ## -------------------------------------------
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
