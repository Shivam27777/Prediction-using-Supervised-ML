# Importing all libraries required 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Reading data from link provided
url = 'http://bit.ly/w-data'
scoreMarks = pd.read_csv(url)
print(scoreMarks.head())

# Plotting the distribution of scores
X = scoreMarks.iloc[:, :-1].values
y = scoreMarks.iloc[:, 1].values
plt.scatter(X, y)
plt.xlabel("No. of hours")
plt.ylabel("Marks scored")
plt.show()

#training with the existing data
reg = LinearRegression()
reg.fit(scoreMarks[['Hours']],scoreMarks['Scores'])

#printing slope and interceot of regression equation
print(reg.coef_)
print(reg.intercept_)

#finding out the score of student who studies 9.25hrs/day 
hours = 9.25
pridted_Score = reg.coef_*hours + reg.intercept_
#printing the score
print(pridted_Score)

#answer is 92.90 marks for this dataset
