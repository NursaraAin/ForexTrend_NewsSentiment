import pandas as pd
import streamlit as st
import textwrap
import plotly.express as px
from datetime import datetime, timedelta
from func_DataProcess import predictWeekly,loadForex,loadNews

current_date = datetime.now()
    
def displayImportance(importance):
    switch = {
        0: ":grey_exclamation:",
        1: ":exclamation:",
        2: ":exclamation::exclamation:",
        3: ":exclamation::exclamation::exclamation:"
    }
    return switch.get(importance, "Invalid importance value")

def displayNews():
    news=loadNews()
    
    monday = current_date - timedelta(days=current_date.weekday()+7)
    next_monday = monday + timedelta(days=6)

    monday_str = monday.strftime('%Y-%m-%d')
    next_monday_str = next_monday.strftime('%Y-%m-%d')
    
    news['Date_Group'] = pd.to_datetime(news['Date']).dt.strftime("%Y-%m-%d")
    news = news.sort_values('Date', ascending = False)
    berita = news[(news['Date_Group'] >= monday_str) & (news['Date_Group'] <= next_monday_str)]
    
    st.markdown("## Current News of the Week (United States and Malaysia)")
    st.markdown(f"### {monday_str} until {next_monday_str}")
    st.markdown("No importance (:grey_exclamation:), "+
                "Low importance (:exclamation:), "+
                "Medium importance (:exclamation::exclamation:), "+
                "High importance(:exclamation::exclamation::exclamation:)")
    count=1
    
    for index,article in berita.iterrows():
        importance = displayImportance(article['importance'])
        st.subheader(f"{count}. {article['title']} [{importance}]")
        formatted_date = pd.to_datetime(article['Date']).strftime("%d %B %Y @%I:%M %p")
        st.caption(f"Date: {formatted_date}")
        description = article['description']
        words = description.split()
        if len(words) > 30:
            with st.expander("Expand to see full description"):
                st.text(textwrap.fill(description, width=60))
        else:
            truncated_text = description
            wrapped_text = textwrap.fill(truncated_text, width=60)
            st.text(wrapped_text)       
            
        #st.markdown('-----')
        count=count+1
  
def displayPrediction():
    data = pd.read_csv('C:/Users/nursa/source/repos/NewsScraping/NewsScraping/datalog.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    last_date =  data['Date'].max()
    
    #if the last date from the datalog has not passed for a week
    if((current_date - last_date).days<7):
        #get the previous week's monday date
        monday = current_date - timedelta(days=current_date.weekday()+7)
        change = False
    #if the last date from the datalog has passed for a week
    else: 
        #get the this week's monday date
        monday = current_date - timedelta(days=current_date.weekday())
        change = True
    
    prev_monday = monday - timedelta(days=7)
    
    monday_str = monday.strftime('%Y-%m-%d')
    prev_monday_str = prev_monday.strftime('%Y-%m-%d')
    
    line = predictWeekly(change,data,monday_str,prev_monday_str)
    
    st.markdown(line)
    st.markdown(f"*Predicted for {monday_str} until {(monday + timedelta(days=6)).strftime('%Y-%m-%d')}")

    
def displayForex():
    forex=loadForex("WEEKLY")
    # start_month = current_date - timedelta(weeks=5)
    # st.header("USD/MYR Forex Prices")
    # forex['Date'] = pd.to_datetime(forex['Date'])
    # df = forex[(forex['Date'] >= start_month) & (forex['Date'] <= current_date)]
    
    fig = px.line(forex, x='Date', y='Close', markers = True)

    # Configure the chart layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Close Price',
    )
    
    # Display the chart in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)
    
def displayTable():
    st.markdown("""
        <style>
        table {
            border-collapse: collapse;
            border-spacing: 0;
        }
        td, th {
            border: none;
            padding: 8px;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)
    data = pd.read_csv('C:/Users/nursa/source/repos/NewsScraping/datalog.csv')
    data = data[['Date','Polarity','Compound']]
    table = data.to_html(index=False, classes='dataframe')
    st.markdown(table, unsafe_allow_html=True)