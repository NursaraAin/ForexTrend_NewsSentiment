import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import json
import requests
import yfinance as yf
from datetime import datetime,timedelta
import textwrap

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


##CHANGE FOLDER###
lda = joblib.load("C:/Users/nursa/source/repos/NewsScraping/linear_disc_analysis.joblib")
current_date = datetime.now()

def calculate4Price(rate_df):
  # Calculate price change
  delta = rate_df['Close'].diff()

  # Calculate gains and losses
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)

  # Calculate average gain and average loss
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()

  # Calculate relative strength
  rs = avg_gain / avg_loss

  # Calculate RSI
  rsi = 100 - (100 / (1 + rs))

  # Add RSI to DataFrame
  rate_df['RSI'] = rsi

  # Calculate the 10-day EMA of the 'Close' column
  ema = rate_df['Close'].ewm(span=10, adjust=False).mean()

  # Add the 'EMA' column to the DataFrame
  rate_df['EMA'] = ema

  # Calculate the highest high and lowest low over a 14-day window
  high = rate_df['Close'].rolling(window=14).max()
  low = rate_df['Close'].rolling(window=14).min()

  # Calculate the %K and %D lines
  k = 100 * ((rate_df['Close'] - low) / (high - low))
  d = k.rolling(window=3).mean()

  # Add the %K and %D columns to the DataFrame
  rate_df['%K'] = k
  rate_df['%D'] = d
  return rate_df

def processText (news_df):
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  clean_pattern = re.compile(r"\r\n")
  table = str.maketrans("", "", string.punctuation)
  clean_headlines = []

  for row in news_df.itertuples():
    combined_text = ' '.join([str(row.title), str(row.description)])
    text = clean_pattern.sub("", combined_text)
    #text = re.sub(r'\b[A-Z]+\b|[^a-zA-Z0-9\s]', '', text)
    sentences = nltk.sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    tokens = [[word.lower() for word in sentence] for sentence in tokens]
    tokens = [lemmatizer.lemmatize(word.translate(table)) for sentence in tokens for word in sentence if word not in stop_words]
    combined_tokens = ' '.join(tokens)
    clean_headlines.append(combined_tokens)

  news_df['Combine_Text'] = clean_headlines
  return news_df

def rakeKeywords(news_df):
  r = Rake()
  # Define a function to extract keywords from text
  def extract_keywords(text):
      r.extract_keywords_from_text(text)  # Extract keywords from text
      keyword_phrases = r.get_ranked_phrases()  # Get ranked keyword phrases
      return keyword_phrases

  # Apply the extract_keywords function to the 'Combine_Text' column
  news_df['rake_keywords'] = news_df['Combine_Text'].apply(extract_keywords)

  return news_df

def nltk_textblob_Sentiment(news_df):
  sia = SentimentIntensityAnalyzer()

  compound_score = []
  txtblob = []
  for row in news_df.itertuples():
    keywords = str(row.rake_keywords)
    tb = TextBlob(keywords)
    txtblob.append(tb.sentiment.polarity)
    compound_score.append(sia.polarity_scores(keywords)['compound'])

  news_df['Polarity'] = txtblob
  news_df['Compound'] = compound_score
  return news_df

def getNews():
    countries={'malaysia','united states'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0'}
    news = []

    for country in countries:
        url = f"https://tradingeconomics.com/ws/stream.ashx?c={country}&start=0&size=100"
        response = requests.request("GET", url, headers=headers)
        if(response.status_code == 200):
            news = news + json.loads(response.text)

    news = pd.DataFrame(news)
    news = news.rename(columns={'date': 'Date'})

    news = nltk_textblob_Sentiment(rakeKeywords(processText(news)))

    news['Date'] = pd.to_datetime(news['Date'], format='%Y-%m-%dT%H:%M:%S')
    # news['Date'] = news['Date'].dt.strftime('%d/%m/%Y')
    # news['Date'] = pd.to_datetime(news['Date'], format='%d/%m/%Y')

    average_sentiment = news.groupby(pd.Grouper(key='Date', freq='D')).mean(numeric_only=True)[['Polarity', 'Compound', 'importance']].dropna()
    average_sentiment['Date'] = average_sentiment.index
    average_sentiment = average_sentiment.reset_index(drop=True)
    return average_sentiment

def getForex():
    forex = yf.download("USDMYR=X", start="2013-01-01", end=current_date)
    forex['Date'] = forex.index
    forex = forex.reset_index(drop=True)
    forex['Date'] = pd.to_datetime(forex['Date'], format='%d/%m/%Y')
    forex = forex.sort_values('Date')
    forex = calculate4Price(forex)
    
    return forex

def getResult(array):
    count_0 = np.count_nonzero(array == 0)  # Count the occurrences of 0
    count_1 = np.count_nonzero(array == 1)  # Count the occurrences of 1
      
    total_elements = array.size  # Total number of elements in the array
    
    percentage_0 = (count_0 / total_elements) * 100
    percentage_1 = (count_1 / total_elements) * 100
    
    if count_0 > count_1:
        predict = 0
        score = round(percentage_0, 2)
        if count_0 == total_elements:
            result = "## All of the news articles for the week forecasted a downward or no movement for the upcoming week. :chart_with_downwards_trend:"
        else:
            result = f"## Approximately {score}% of the news articles for the week anticipate a downward or no movement in the upcoming week. :chart_with_downwards_trend:"
    elif count_1 > count_0:
        predict = 1
        score = round(percentage_1, 2)
        if count_1 == total_elements:
            result = "## All of the news articles for the week forecasted an upward movement for the upcoming week. :chart_with_upwards_trend:"
        else:
            result = f"## Approximately {round(percentage_1, 2)}% of the news articles for the week anticipate a upward movement in the upcoming week. :chart_with_upwards_trend:"
    else:
        result = "## All of the news articles for the week forecasted both trends movement for upcoming week :confused:"
        predict = 0
        score = 50.0
    
    return result,predict,score

def increase_decrease(df, col_name):
    results = []
    for i in range(len(df) - 1):
        if df[col_name][i] < df[col_name][i+1]:
            results.append(1)
        elif df[col_name][i] > df[col_name][i+1]:
            results.append(0)
        else:
            results.append(1)
    results.append(1)
    df["next_trend"] = results
    return df

def getActual(prev_monday_str,monday_str):
    weeklyfx = yf.download("USDMYR=X", start=prev_monday_str, end=monday_str, interval="1wk")
    if(len(weeklyfx) < 2):
        prev_monday = datetime.strptime(prev_monday_str, '%Y-%m-%d') - timedelta(days=1)
        prev_monday_str = prev_monday.strftime('%Y-%m-%d')
        weeklyfx = yf.download("USDMYR=X", start=prev_monday_str, end=monday_str, interval="1wk")
    actual = 1 if weeklyfx['Close'].iloc[0] < weeklyfx['Close'].iloc[1] else 0
    return actual

def getNewData(monday_str,prev_monday_str):
    news = getNews()
       
    news_filtered = news[(news['Date'] <= monday_str) & (news['Date'] >= prev_monday_str)]
    
    forex = getForex()
    
    df = pd.merge(news_filtered, forex, on='Date', how='inner')
    keep_columns=['Date','Polarity','Compound','importance','Open','High','Low','Close','RSI','EMA','%K','%D']
    df=df[keep_columns]
    df=df.fillna(0)
    return df

def displayImportance(importance):
    switch = {
        0: ":grey_exclamation:",
        1: ":exclamation:",
        2: ":exclamation::exclamation:",
        3: ":exclamation::exclamation::exclamation:"
    }
    return switch.get(importance, "Invalid importance value")

def loadApp():
##CHANGE FOLDER###
    past = pd.read_csv("C:/Users/nursa/source/repos/NewsScraping/saved.csv")
    last_date = datetime.strptime(past['PredictFor_Date'].max(),'%Y-%m-%d')
    
    if((current_date - last_date).days>=7):
        monday = current_date - timedelta(days=current_date.weekday())
        prev_monday = monday - timedelta(days=7)
        monday_str = monday.strftime('%Y-%m-%d')
        prev_monday_str = prev_monday.strftime('%Y-%m-%d')
        
        df = getNewData(monday_str,prev_monday_str)
        X=np.array(df.drop(['Date'],axis=1))
        lda_predicts=lda.predict(X)
        df['Predicted']=lda_predicts
##CHANGE FOLDER###
        df.to_csv("C:/Users/nursa/source/repos/NewsScraping/daily_predict.csv", mode='a', header=False, index=False)
        
        result,predict,score = getResult(lda_predicts)
        actual = getActual(prev_monday_str, monday_str)
        
        past.loc[past.index[-1], 'Actual'] = actual
        
        save_result = {'Date':[prev_monday_str],
                       'PredictFor_Date':[monday_str],
                       'Result':[result],
                       'Predict':[predict],
                       'Score_R':[score],
                       'Actual':[-1]}
        
        past = pd.concat([past,pd.DataFrame(save_result)])
##CHANGE FOLDER###
        past.to_csv("C:/Users/nursa/source/repos/NewsScraping/saved.csv",index=False)

def displayNews():
    countries={'malaysia','united states'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0'}
    news = []

    for country in countries:
        url = f"https://tradingeconomics.com/ws/stream.ashx?c={country}&start=0&size=5"
        response = requests.request("GET", url, headers=headers)
        if(response.status_code == 200):
            news = news + json.loads(response.text)

    news = pd.DataFrame(news)
    news = news.rename(columns={'date': 'Date'})

    # monday = current_date - timedelta(days=current_date.weekday())
    # next_monday = monday + timedelta(days=6)

    # monday_str = monday.strftime('%Y-%m-%d')
    # next_monday_str = next_monday.strftime('%Y-%m-%d')

    # news['Date_Group'] = pd.to_datetime(news['Date']).dt.strftime("%Y-%m-%d")
    news = news.sort_values('Date', ascending = False)
    # berita = news[(news['Date_Group'] >= monday_str) & (news['Date_Group'] <= next_monday_str)]

    st.markdown("## Current News of the Week (United States and Malaysia)")
    #st.markdown(f"### {monday_str} until {next_monday_str}")
    st.markdown("No importance (:grey_exclamation:), "+
                "Low importance (:exclamation:), "+
                "Medium importance (:exclamation::exclamation:), "+
                "High importance(:exclamation::exclamation::exclamation:)")
    #count=1

    for index,article in news.iterrows():
        importance = displayImportance(article['importance'])
        st.subheader(f"{article['title']} [{importance}]")
        formatted_date = pd.to_datetime(article['Date']).strftime("%d %B %Y @%I:%M %p")
        st.caption(f"Date: {formatted_date}")
        description = article['description']
        words = description.split()
        if len(words) > 30:
            with st.expander("Expand to see full description"):
                st.text(textwrap.fill(description, width=90))
        else:
            truncated_text = description
            wrapped_text = textwrap.fill(truncated_text, width=60)
            st.text(wrapped_text)       
            
        #st.markdown('-----')
        #count=count+1

import streamlit as st
import plotly.graph_objects as go

loadApp()
start_month = current_date - timedelta(weeks=5)
weeklyfx = yf.download("USDMYR=X", start=start_month.strftime('%Y-%m-%d'), end=current_date.strftime('%Y-%m-%d'),interval="1wk")
###CHANGE FOLDER###
past = pd.read_csv("C:/Users/nursa/source/repos/NewsScraping/saved.csv")
past['Date'] = pd.to_datetime(past['Date'],format = '%Y-%m-%d')

data = pd.merge(past, weeklyfx, on='Date', how='inner')
# data = weeklyfx.merge(data, on='Date', how='left')
# data = data.fillna("")

# Add arrows based on the next_trend values
# symbol_map = {
#     (1, -1): ('triangle-up', 'blue'),
#     (1, 0): ('triangle-up', 'red'),
#     (1, 1): ('triangle-up', 'green'),
#     (0, -1): ('triangle-down', 'blue'),
#     (0, 0): ('triangle-down', 'green'),
#     (0, 1): ('triangle-down', 'red')
# }

fig = go.Figure()

# Add line chart
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='lines',
    name='Close Price',
))

# Add markers
for i in range(len(data)):
    predicted = data['Predict'][i]
    actual = data['Actual'][i]
    
    marker_color = 'blue'
    marker_symbol = 'circle'
    shw = False
    
    if predicted in [0, 1] and actual in [-1, 0, 1]:
        marker_symbol = 'triangle-up' if predicted == 1 else 'triangle-down'
        marker_color = 'green' if predicted == actual else 'blue' if actual == -1 else 'red'
        shw = True
    
    fig.add_annotation(
        x=data['Date'][i],
        y=data['Close'][i],
        text='',
        showarrow=shw,
        arrowhead=0,
        arrowsize=4,
        arrowwidth=2,
        arrowcolor=marker_color,
        ax=0,
        ay=-30 if predicted == 0 else 30
    )

    fig.add_trace(go.Scatter(
        x=[data['Date'][i]],
        y=[data['Close'][i]],
        mode='markers',
        marker=dict(symbol=marker_symbol, color=marker_color, size=10),
    ))
    
# Hide Legends
fig.update_traces(showlegend=False)
# Customize the chart
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Close Price'),
)

st.set_page_config(layout="wide")
div1,div2 = st.columns([3,2], gap="medium")
# Display the chart in Streamlit
with div1:
    st.markdown("## Weekly Forex Price with Predicted Up/Down Trend")
    st.plotly_chart(fig,use_container_width=True)
with div2:
    st.markdown(data['Result'].iloc[-1])
    st.markdown('------')
    st.markdown('The <span style="color:blue">blue</span> arrow represents the predicted trend of the closing price', unsafe_allow_html=True)
    st.markdown('If the prediction is correct, the arrow will turn <span style="color:green">green</span>, otherwise it will turn <span style="color:red">red</span>', unsafe_allow_html=True)

displayNews()


    
    
    
    



