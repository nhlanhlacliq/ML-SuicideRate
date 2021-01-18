# Imports
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv

# Read data from csv function. Return numpy array of data
def read_data(path):
	with open (path, 'r') as file:
		csv_reader = csv.reader(file, delimiter=',')
		return np.array(list(csv_reader)[1:])

# Create data set. Choose specifically South African entries from dataset, on specific columns -Country, Year and Suicide Count
dataset = read_data('master.csv')
dataset = dataset[23288:23528, [0,1,4]]

# Modify Dataset. recreate it so it contains the year and the total number of suicides for that year
# Originally, no. of suicides are partitioned in different times of the year. 
# Add the suicide count(entry[2]) to the total for each year(entry[1])
modified_dataset = []
for year in range(1996, 2016):
  year_total = 0
  for entry in dataset:
    if int(entry[1]) == year:
      year_total += int(entry[2])
  modified_dataset.append(([year], [year_total]))
modified_dataset = np.array(modified_dataset)

# Extract data x and y values. Sort, then scatter data points
data_x = np.array(modified_dataset[:,0], dtype = np.int)
data_x = data_x[np.argsort(data_x[:, 0])]
data_y = np.array(modified_dataset[:,1], dtype = np.int)
data_y = data_y[np.argsort(data_y[:, 0])]
plt.grid(True)
plt.scatter(data_x, data_y, c='r')

# Split data into training and testing
data_xTrain = data_x[:-4] 
data_xTest = data_x[-4:]
data_yTrain = data_y[:-4]
data_yTest = data_y[-4:]

# Create polynomial function object with a degree of 2
poly = PolynomialFeatures(degree=2)
# Apply polynomial function on data_x
poly_x_train = poly.fit_transform(data_x)
poly_x_test = poly.transform(data_xTest)

# Create prediction model, fit with polynomial x and y data 
predict_model = LinearRegression()
predict_model.fit(poly_x_train, data_y)

# Use predictionn model to predict from polynomial x data 
y_predict = predict_model.predict(poly_x_train)

# Plot polynomial regression line
plt.plot(data_x, y_predict,"b-", c='g', linestyle='--')

# Customise plot
plt.title('Suicide rate in South Africa')
plt.xlabel('Year')
plt.ylabel('Suicides')
plt.xticks(range(1996,2016,2))
plt.yticks(range(100,600,50))

plt.show()

"""
Data obtained from:
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
"""