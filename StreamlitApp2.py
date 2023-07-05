import streamlit as st
import matplotlib.pyplot as plt
from TheWholeThing import *

forex = getForex()
forex = 
# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 4, 8]

# Create a line chart with up and down symbols
fig, ax = plt.subplots()
ax.plot(x, y, marker='o', linestyle='-')

# Add up or down symbols based on the slope of the line
for i in range(1, len(x)):
    if y[i] > y[i-1]:
        ax.plot(x[i], y[i], marker='^', color='green', markersize=10)
    elif y[i] < y[i-1]:
        ax.plot(x[i], y[i], marker='v', color='red', markersize=10)

# Customize the chart
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Line Chart with Up/Down Symbols')

# Display the chart in Streamlit
st.pyplot(fig)

