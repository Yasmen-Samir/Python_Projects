import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split

# import file with data
car_data = pd.read_csv("car_data.csv")

def data_testing_charts():

    #plotting correlation heatmap
    sb.heatmap(car_data.corr())
    # displaying heatmap
    mp.show() 

    #plotting pairplot 
    # sb.pairplot(car_data)
    # displaying heatmap
    # mp.show()

# displaying features correlation with numbers 
def data_testing_numbers():
    mp.figure(figsize=(12,10))
    cor = car_data.corr()
    sb.heatmap(cor, annot=True, cmap=mp.cm.Reds)
    mp.show()                                          #enginesize // curbweight // citympg // highwaympg


class linear_regression():

    def __init__(self, learning_rate, num_of_iterations):
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        iterations = np.empty(self.num_of_iterations, dtype=object)
        costs = np.empty(self.num_of_iterations, dtype=object)

        # gradient descent
        for i in range(self.num_of_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            iterations[i]=i
            costs[i] = np.mean((y - y_predicted) ** 2)
            
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return iterations, costs


    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


data_testing_charts()
data_testing_numbers()

car_data.drop_duplicates(inplace=True)

car_data = car_data.sample(frac=1)
inputData= pd.DataFrame(car_data, columns=['enginesize', 'curbweight','citympg','highwaympg'])
outputData= car_data["price"]

normalized_data=(inputData-np.min(inputData))/(np.max(inputData)-np.min(inputData))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

X_train, X_test, y_train, y_test = train_test_split(normalized_data, outputData, test_size=0.2, random_state=1234)

rate = [0.001,0.002,0.003]
itnum = [1000,2000,500]

for i in range(len(rate)):
    for j in range(len(itnum)):
        regressor = linear_regression(rate[i], itnum[j])
        iterations,cost = regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        MSE = mean_squared_error(y_test, predictions)
        print("MSE:", MSE)

        acc = accuracy(y_test, predictions)
        print("Accuracy:", acc)

        mp.scatter(iterations,cost,color="blue",linewidth=0)
        mp.title('Cost along each iteration')
        mp.show()
 