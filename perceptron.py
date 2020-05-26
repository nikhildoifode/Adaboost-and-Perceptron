import numpy as np
import csv
import sys

class Perceptron:
    def __init__ (self, n_input, epochs, learning_rate):
        self.w = np.zeros(n_input + 1)
        self.w[0] = 1 #Bias Term
        self.epochs = epochs
        self.lr = learning_rate
        self.error = None

    def model(self, x):
        sum = np.dot(self.w[1:], x)
        if (sum >= self.w[0]):
            return 1
        else:
            return 0

    def predict(self, X):
        Y = []
        for x in X: Y.append(self.model(x))

        return np.array(Y)

    def trainError(self, X, Y):
        for x, y in zip(X, Y):
            y_pred = self.model(x)
            if y != y_pred:
                self.w[1:] += self.lr * (y - y_pred) * x
                self.w[0] += self.lr * (y_pred - y)

    def fit(self, X, Y):
        max_accuracy = 0
        weightVector = None
        prevAcuracy = float('inf')
        count = 0

        for _ in range(self.epochs):
            self.trainError(X, Y)
            accuracy = calculate_accuracy(self.predict(X), Y)
            if (accuracy > max_accuracy):
                max_accuracy = accuracy
                weightVector = self.w
            if (abs(prevAcuracy - accuracy) < 0.0001):
                if (count == 5):
                    break
                count += 1
            else: count = 0
            prevAcuracy = accuracy

        self.w = weightVector
        self.error = 1 - max_accuracy

def calculate_accuracy(label, prediction):
    return np.mean(label == prediction)

def shuffle_data(X, Y):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    return X[index], Y[index]

def main ():
    if len(sys.argv) < 5 or len(sys.argv) > 6 or '--dataset' not in sys.argv or '--mode' not in sys.argv:
        print("Make sure command is of type: " +
        "python3 perceptron.py --dataset /path/to/data/filename.csv --mode <erm|cv> <folds(cv_mode_only)>")
        return

    filePath = sys.argv[sys.argv.index('--dataset') + 1]
    mode = sys.argv[sys.argv.index('--mode') + 1]

    try:
        with open(filePath)	as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            data_list.pop(0)

            X = np.array(data_list, dtype = np.float64)
            Y = X[:, X.shape[1] - 1]
            Y = Y.astype(int)
            X = X[:, :-1]

            if (mode == 'erm'):
                perceptron = Perceptron(X.shape[1], 1000, 0.1)
                perceptron.fit(X, Y)

                print("Training Error: ", perceptron.error)
                print("Weight Vector: ", perceptron.w)

            elif (mode == 'cv'):
                X, Y = shuffle_data(X, Y)
                if (len(sys.argv) != 6):
                    print("Make sure command is of type: " +
                    "python3 perceptron.py --dataset /path/to/data/filename.csv --mode cv <folds>")
                    return

                folds = int(sys.argv[sys.argv.index('--mode') + 2])
                if (folds < 2 or folds > 20):
                    print("value for number of folds too high or too low. Try between [2,20]")
                    return

                fold_size = int(X.shape[0] / folds)
                perceptron = {}

                for i in range(folds):
                    X_1 = X[:i*fold_size,:]
                    X_2 = X[(i+1)*fold_size:,:]
                    Y_1 = Y[:i*fold_size]
                    Y_2 = Y[(i+1)*fold_size:]
                    if (X_1.shape[0] == 0):
                        X_train = X_2
                        Y_train = Y_2
                    elif (X_2.shape[0] == 0):
                        X_train = X_1
                        Y_train = Y_1
                    else:
                        X_train = np.concatenate((X_1, X_2), axis=0)
                        Y_train = np.concatenate((Y_1, Y_2), axis=0)

                    X_cv = X[i*fold_size:(i+1)*fold_size,:]
                    Y_cv = Y[i*fold_size:(i+1)*fold_size]

                    perceptron[i] = Perceptron(X_train.shape[1], 1000, 0.1)
                    perceptron[i].fit(X_train, Y_train)
                    perceptron[i].error = 1 - calculate_accuracy(perceptron[i].predict(X_cv), Y_cv)

                minError = 1.0
                iter = 0
                sum = 0
                for i in perceptron:
                    print("Error of fold ", (i+1), perceptron[i].error)
                    print("Weight Vector of fold ",(i+1), ": ", perceptron[i].w)
                    sum += perceptron[i].error
                    if (perceptron[i].error < minError):
                        iter = i
                        minError = perceptron[i].error

                print("Mean Error: ", sum / folds)
                print("Training Error  for Optimal Hypothesis: ", perceptron[iter].error)
                print("Weight Vector for Optimal Hypothesis: ", perceptron[iter].w)

            else:
                print("For ERM input should be like: " +
                "python3 perceptron.py --dataset /path/to/data/filename.csv --mode erm")
                print("For Cross validation input should be like: " +
                "python3 perceptron.py --dataset /path/to/data/filename.csv --mode cv <folds>")

    except IOError as e:
        print("Couldn't open the file (%s). Check file path, permission and name again." % e)

if __name__ == "__main__":
    main()