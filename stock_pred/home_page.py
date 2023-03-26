import time
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title = "Multipage App",
    page_icon = "🔥",
)


st.title(":blue[Stock Trend Predictor]")

with st.sidebar:
    with st.spinner("Loading..."):
        time.sleep(3)
    st.success("Done!")

st.sidebar.success("Select a page from above.")





st.subheader(":blue[What Is Trend Analysis?]")

st.write("Trend analysis is a technique used in technical analysis that" + 
        "attempts to predict future stock price movements based on recently observed trend data. Trend analysis uses historical data," +
        "such as price movements and trade volume, to forecast the long-term direction of market sentiment.")


img = Image.open("/Users/arunkumar/Desktop/Projects/pages/images/trend_analysis.png")
st.image(img)

st.subheader(":blue[Understanding Trend Analysis]")

st.write("""Trend analysis tries to predict a trend, such as a bull market run, and ride that trend until data suggests a trend reversal, such as a bull-to-bear market. Trend analysis is helpful because moving with trends, and not against them, will lead to profit for an investor. It is based on the idea that what has happened in the past gives traders an idea of what will happen in the future. There are three main types of trends:\n1. Upward\n 2. Downward \n 3. Sideways""")

st.subheader(":blue[Types of Trends to Analyze]")

st.write("""1.Upward trend: \n\t An upward trend, also known as a bull market, is a sustained period of rising prices in a particular security or market.
          Upward trends are generally seen as a sign of economic strength and can be driven by factors such as strong demand, rising profits, and favorable economic conditions.""")

st.write("""2.Downward trend: \n\t A downward trend, also known as a bear market,
 is a sustained period of falling prices in a particular security or market.
   Downward trends are generally seen as a sign of economic weakness and can be driven by factors such as weak demand, declining profits, and unfavorable economic conditions.""")

st.write("""3.Sideways trend: \n\t A sideways trend, also known as a rangebound market,
 is a period of relatively stable prices in a particular security or market.
   Sideways trends can be characterized by a lack of clear direction, with prices fluctuating within a relatively narrow range.""")


st.subheader(":blue[Trend Trading Strategy used:]")

st.write("""Moving Averages: These strategies involve entering into long positions when
a short-term moving average crosses above a long-term moving average, and entering short positions when a short-term moving average crosses below a long-term moving average.
""")

img1 = Image.open("/Users/arunkumar/Desktop/Projects/pages/images/MA.png")

st.image(img1)