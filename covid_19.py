# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set display limit
pd.options.display.max_rows = 100

# Importing Dataset
covid_19 = pd.read_csv('./covid_19.csv', 
                       parse_dates=['Date'])

# Selecting desired columns
covid_19 = covid_19[['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

# Create a synthetic paramter, Nth_day
covid_19['Nth_day'] = (covid_19['Date'] - min(covid_19['Date'])).dt.days

print(covid_19)

