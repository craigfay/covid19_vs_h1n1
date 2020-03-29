# Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set display limit
pd.options.display.max_rows = 100

# Suppress scientific notation
np.set_printoptions(suppress=True)


# COVID-19
# --------

# Importing Dataset
c19 = pd.read_csv('./covid_19.csv', parse_dates=['Date'])

# Selecting desired columns
c19 = c19[['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

# Create a synthetic paramter, Nth_day
c19['Nth_day'] = (c19['Date'] - min(c19['Date'])).dt.days

# Collapse all records that have the same nth_day onto one record
c19_total_days = c19['Nth_day'].at[len(c19['Nth_day']) -1]
# print(c19.tail(1)['Nth_day'])
#c19_total_days = c19.tail(1)['Nth_day'][0]
data = np.zeros(shape=(c19_total_days + 1, 2))
for n, row in c19.iterrows():
    nth_day = row['Nth_day']
    data[nth_day][0] += row['Confirmed']
    data[nth_day][1] += row['Deaths']
c19 = pd.DataFrame(data=data, columns=['confirmed', 'deaths'])
c19_first_n_days = np.arange(0, c19_total_days + 1) 


# H1N1
# ----

# Importing Dataset
h1n1 = pd.read_csv('./h1n1.csv', parse_dates=['Update Time'], encoding='ISO-8859-1')

# Create a synthetic paramter, Nth_day
h1n1['Nth_day'] = (h1n1['Update Time'] - min(h1n1['Update Time'])).dt.days
h1n1 = h1n1.sort_values(by='Nth_day')

# Collapse all records that have the same nth_day onto one record
h1n1_total_days = h1n1.tail(1)['Nth_day'][0]
data = np.zeros(shape=(h1n1_total_days + 1, 2))
for n, row in h1n1.iterrows():
    nth_day = row['Nth_day']
    data[nth_day][0] += row['Cases']
    data[nth_day][1] += row['Deaths']
h1n1 = pd.DataFrame(data=data, columns=['confirmed', 'deaths'])
h1n1_first_n_days = np.arange(0, h1n1_total_days + 1) 

# Filter out missing values
h1n1 = h1n1[h1n1['confirmed'] > 0]
h1n1 = h1n1[h1n1['deaths'].notnull()]


# Visualization
# ------------

plt.plot(c19['confirmed'], '--', color='red', label='C19 Cases')
plt.plot(c19['deaths'], color='red', label='C19 Deaths')

plt.plot(h1n1['confirmed'], '--', color='blue', label='H1N1 Cases')
plt.plot(h1n1['deaths'], color='blue', label='H1N1 Deaths')

plt.xlabel('First N Days')
plt.ylabel('Population')
plt.title('COVID-19')
plt.legend()
plt.show()

