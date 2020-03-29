# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set display limit
pd.options.display.max_rows = 100

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Importing Dataset
covid_19 = pd.read_csv('./covid_19.csv', 
                       parse_dates=['Date'])

# Selecting desired columns
covid_19 = covid_19[['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]

# Create a synthetic paramter, Nth_day
covid_19['Nth_day'] = (covid_19['Date'] - min(covid_19['Date'])).dt.days


# Collapse all records that have the same nth_day onto one record
total_days = covid_19['Nth_day'].at[len(covid_19['Nth_day']) -1]
data = np.zeros(shape=(total_days + 1, 3))
for n, row in covid_19.iterrows():
    nth_day = row['Nth_day']

    data[nth_day][0] += row['Confirmed']
    data[nth_day][1] += row['Deaths']
    data[nth_day][2] += row['Recovered']
c19 = pd.DataFrame(data=data, columns=['confirmed', 'deaths', 'recovered'])

first_n_days = np.arange(0, total_days + 1) 

# COVID-19
plt.plot(first_n_days, c19['confirmed'], '--', color='red', label='Confirmed Cases')
plt.plot(first_n_days, c19['deaths'], color='red', label='Deaths')
plt.xlabel('First N Days')
plt.ylabel('Population')
plt.title('COVID-19')
plt.legend()
plt.show()

