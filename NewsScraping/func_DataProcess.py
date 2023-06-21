import pandas as pd
import numpy as np
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json
import os
import joblib
from datetime import date,datetime,timedelta
import yfinance as yf
#523,42

lda = joblib.load("C:/Users/nursa/source/repos/NewsScraping/NewsScraping/linear_disc_analysis.joblib")

current_date = datetime.now()

# monday=datetime(2023,5,29)
# next_monday=datetime(2023,6,5) 

# monday_str = monday.strftime('%Y-%m-%d')
# next_monday_str = next_monday.strftime('%Y-%m-%d')

def loadNews():
    countries={'malaysia','united states'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0'}
    news = []
    for country in countries:
        url = f"https://tradingeconomics.com/ws/stream.ashx?c={country}&start=0&size=100"
        response = requests.request("GET", url, headers=headers)
        if(response.status_code == 200):
            news = news + json.loads(response.text)
        # else:
        #     return f"Response code {response.status_code}"
    #news_df = pd.read_csv("C:/Users/nursa/source/repos/NewsScraping/NewsData_TE_2905-130623.csv")
    news_df = pd.DataFrame(news)
    news_df = news_df.rename(columns={'date':'Date'})
    news_df['Date'] = pd.to_datetime(news_df['Date'], format='%Y-%m-%d %H:%M:%S')
    #news_df['Date'] = pd.to_datetime(news_df['Date'], format='%d/%m/%Y %H:%M')
    news_df['Date'] = news_df['Date'].dt.strftime('%d/%m/%Y')
    news_df['Date'] = pd.to_datetime(news_df['Date'], format='%d/%m/%Y')
    return news_df

def loadForex(jenis):
    pair = "USDMYR=X"
    if jenis == "DAILY":
        forex = yf.download(pair, start="2013-01-01", end=current_date)
    elif jenis == "WEEKLY":
        # start_date = (date(current_date.year, 1, 1)).strftime("%Y-%m-%d")
        forex = yf.download(pair, start="2013-01-01", end=current_date,  )
    # url = f"https://www.alphavantage.co/query?function=FX_{jenis}&from_symbol=USD&to_symbol=MYR&apikey=G3MXR71ZPWV7LU62"
    # response = requests.request("GET", url)
    # if(response.status_code == 200):
    #     forex = json.loads(response.text)
    #     col = f"Time Series FX ({jenis.capitalize()})"
    #     forex = pd.DataFrame.from_dict(forex[col], orient='index')
    #     forex = forex.rename(columns={'1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close'})
    
    forex['Date'] = forex.index
    forex = forex.reset_index(drop=True)
    forex['Date'] = pd.to_datetime(forex['Date'], format='%d/%m/%Y')
    forex = forex.sort_values('Date')
    forex = calculate4Price(forex)
    return forex

def result(array):
    count_0 = np.count_nonzero(array == 0)  # Count the occurrences of 0
    count_1 = np.count_nonzero(array == 1)  # Count the occurrences of 1
      
    total_elements = array.size  # Total number of elements in the array
    
    percentage_0 = (count_0 / total_elements) * 100
    percentage_1 = (count_1 / total_elements) * 100
    
    if count_0 == total_elements:
        result = "# All of the news articles for the week forecasted a downward or no movement for the upcoming week. :chart_with_downwards_trend:"
    elif count_1 == total_elements:
        result = "# All of the news articles for the week forecasted an upward movement for the upcoming week. :chart_with_upwards_trend:"
    elif count_0 > count_1:
        result = f"# Approximately {round(percentage_0, 2)}% of the news articles for the week anticipate a downward or no movement in the upcoming week. :chart_with_downwards_trend:"
    elif count_1 > count_0:
        result = f"# Approximately {round(percentage_1, 2)}% of the news articles for the week anticipate a upward movement in the upcoming week.	:chart_with_upwards_trend:"
    else:
        result = "# All of the news articles for the week forecasted both trends movement for upcoming week :confused:"
    return result

def predictWeekly(passed,data,monday_str,prev_monday_str):
    if(passed):
        news=loadNews()
        news['Date'] = pd.to_datetime(news['Date'])
        news = news[(news['Date'] <= monday_str) & (news['Date'] >= prev_monday_str)]
        news = nltk_textblob_Sentiment(rakeKeywords(processText(news)))
        
        forex = loadForex("DAILY")
        
        df = pd.merge(news, forex, on='Date', how='inner')
        keep_columns=['Date','Polarity','Compound','importance','Open','High','Low','Close','RSI','EMA','%K','%D']
        df=df[keep_columns]
        df=df.fillna(0)
        df = pd.concat([data,df], axis=0)
        X=np.array(df.drop(columns='Date'))
    else:
        df = data
        df.drop_duplicates()
        X=np.array(df.drop(columns=['Date','Predicted','Actual']))
    
    lda_predicts=lda.predict(X)
    df['Predicted'] = lda_predicts
    df = increase_decrease(df, "Close")
    df.to_csv('datalog.csv',index=False)
    df_weekly = df[(df['Date'] < monday_str) & (df['Date'] >= prev_monday_str)]
    
    line = f"Accuracy of model so far: {round((df['Predicted'] == df['Actual']).mean())}\n" + result(df_weekly['Predicted'])
    return line
    
def calculate4Price(rate_df):
    rate_df['Close'] = pd.to_numeric(rate_df['Close'], errors='coerce')
    
    # Calculate price change
    delta = rate_df['Close'].diff()
    
    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Add RSI to DataFrame
    rate_df['RSI'] = rsi
    
    # Calculate the 10-day EMA of the 'Close' column
    ema = rate_df['Close'].ewm(span=5, adjust=False).mean()
    
    # Add the 'EMA' column to the DataFrame
    rate_df['EMA'] = ema
    
    # Calculate the highest high and lowest low over a 7-day window
    high = rate_df['Close'].rolling(window=7).max()
    low = rate_df['Close'].rolling(window=7).min()
    
    # Calculate the %K and %D lines
    k = 100 * ((rate_df['Close'] - low) / (high - low))
    d = k.rolling(window=3).mean()
    
    # Add the %K and %D columns to the DataFrame
    rate_df['%K'] = k
    rate_df['%D'] = d

    return rate_df

def increase_decrease(df, col_name):
    results = []
    for i in range(len(df) - 1):
        if df[col_name][i] < df[col_name][i+1]:
            results.append(1)
        elif df[col_name][i] > df[col_name][i+1]:
            results.append(0)
        else:
            results.append(1)
    results.append(-1)
    df["Actual"] = results
    return df

def processText (news_df):
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  clean_pattern = re.compile(r"\r\n")
  table = str.maketrans("", "", string.punctuation)
  clean_headlines = []

  for row in news_df.itertuples():
    combined_text = ' '.join([str(row.title), str(row.description)])
    text = clean_pattern.sub("", combined_text)
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
  # get mean of polarity, compound, and importance for each day
  news_df = news_df.groupby(pd.Grouper(key='Date', freq='D'))[['Polarity', 'Compound', 'importance']].mean().dropna()
  return news_df