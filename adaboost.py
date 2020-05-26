import numpy as np
import csv
import sys
import operator
import math
import matplotlib.pyplot as plt

class DecisionStump():
    def __init__ (self):
        self.sign = None
        self.feature_index = None
        self.threshold = None
        self.wt = None

class AdaBoost:
    def __init__ (self, n_input, epochs):
        self.Dt = np.ones(n_input)
        self.Dt = self.Dt / n_input
        self.error = None
        self.epochs = epochs

    def model(self, X, Y, stump):
        predictions = np.ones(Y.shape)
        predictions[np.where(X[:,stump.feature_index] < stump.threshold)[0]] = -1 * stump.sign
        predictions[np.where(X[:,stump.feature_index] >= stump.threshold)[0]] = 1 * stump.sign
        return predictions

    def trainError(self, X, Y, feature_i):
        X_Y_D = np.array(sorted(zip(X[:,feature_i], Y, self.Dt), key=operator.itemgetter(0)))
        XSorted = X_Y_D[:,0]
        YSorted = X_Y_D[:,1]
        DSorted = X_Y_D[:,2]
        XIndexes = np.where(XSorted[:-1] < XSorted[1:])[0]
        if len(XIndexes) <= 0:
            return 0.5, 0, 1.0

        #defining boundary
        Y_D = YSorted * DSorted
        cumsum_inc = np.cumsum(Y_D[::1])
        cumsum_dec = np.cumsum(Y_D[::-1])
        cumsum_final = cumsum_dec[-1:0:-1] - cumsum_inc[0:-1:1]

        #find pivot between the boundary
        max_ind, max_score = max(zip(XIndexes, abs(cumsum_final[XIndexes])), key=operator.itemgetter(1))
        error = 0.5 - (0.5 * max_score)
        threshold = (XSorted[max_ind] + XSorted[max_ind+1]) / 2
        sign = np.sign(cumsum_final[max_ind])

        return error, threshold, sign

    def fit(self, X, Y):
        n_features = X.shape[1]
        y_pred = np.zeros(X.shape[0])

        self.stumps = []
        errors = []
        for _ in range(self.epochs):
            stump = DecisionStump()
            epsilon = float('inf')
            for feature_i in range(n_features):
                error, threshold, sign = self.trainError(X, Y, feature_i)
                if error < epsilon:
                    epsilon = error
                    stump.sign = sign
                    stump.threshold = threshold
                    stump.feature_index = feature_i

            stump.wt = 0.5 * math.log((1.0 - epsilon) / epsilon)
            predictions = self.model(X, Y, stump)

            self.Dt *= np.exp(-stump.wt * Y * predictions)
            self.Dt /= np.sum(self.Dt)

            y_pred += stump.wt * predictions
            errors.append(np.sum(np.sign(y_pred) != Y) / Y.shape[0])
            self.stumps.append(stump)

        return np.sign(y_pred), np.array(errors)

    def predict(self, X, Y_Test):
        y_pred = np.zeros(X.shape[0])
        errors = []
        for stump in self.stumps:
            predictions = self.model(X, Y_Test, stump)
            y_pred += stump.wt * predictions
            errors.append(np.sum(np.sign(y_pred) != Y_Test) / Y_Test.shape[0])

        y_pred = np.sign(y_pred)
        return y_pred, np.array(errors)

def calculate_accuracy(label, prediction):
    return np.mean(label == prediction)

def shuffle_data(X, Y):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    return X[index], Y[index]

def main():
    if len(sys.argv) < 5 or len(sys.argv) > 7 or '--dataset' not in sys.argv or '--mode' not in sys.argv:
        print("Make sure command is of type: " +
        "python3 adaboost.py --dataset /path/to/data/filename.csv --mode <erm|cv> <folds(cv_mode_only)>")
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
            Y[Y == 0] = -1

            if (mode == 'erm'):
                wt_req=""
                if (len(sys.argv) == 6):
                    wt_req = sys.argv[sys.argv.index('--mode') + 2]

                adaboost = AdaBoost(X.shape[0], 13)
                y_pred, train_error = adaboost.fit(X, Y)

                accuracy = calculate_accuracy(y_pred, Y)
                adaboost.error = 1 - accuracy

                print("Training Error: ", adaboost.error)
                if (wt_req == "wp"): print("Weight Vector: ", adaboost.Dt)

                plt.figure()
                plt.scatter(range(0,13), train_error, marker="x", label="Training error")
                plt.title("Adaboost on Entire Trainting Set")
                plt.xlabel("Number of iterations (T)")
                plt.ylabel("Error")
                plt.legend()
                if (wt_req == "wp"): plt.show()

            elif (mode == 'cv'):
                folds = int(sys.argv[sys.argv.index('--mode') + 2])
                if (folds < 2 or folds > 13):
                    print("value for number of folds too high or too low. Try between [2,13]")
                    return

                wt_req=""
                if (len(sys.argv) == 7):
                    wt_req = sys.argv[sys.argv.index('--mode') + 3]

                X, Y = shuffle_data(X, Y)
                fold_size = int(X.shape[0] / folds)
                adaboost = {}
                test_errors = np.zeros(13)
                train_errors = np.zeros(13)

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

                    adaboost[i] = AdaBoost(X_train.shape[0], 13)
                    _, train_error = adaboost[i].fit(X_train, Y_train)
                    train_errors += train_error

                    y_pred, test_error = adaboost[i].predict(X_cv, Y_cv)
                    test_errors += test_error
                    adaboost[i].error = 1 - calculate_accuracy(y_pred, Y_cv)

                minError = 1.0
                iter = 0
                sum = 0
                for i in adaboost:
                    print("Error of fold ", (i+1), adaboost[i].error)
                    if (wt_req == "wp"): print("Weight Vector for fold ", (i+1), ": ", adaboost[i].Dt)
                    sum += adaboost[i].error
                    if (adaboost[i].error < minError):
                        iter = i
                        minError = adaboost[i].error

                print("Mean Error: ", sum / folds)
                print("Training Error for Optimal Hypothesis: ", adaboost[iter].error)
                if (wt_req == "wp"): print("Weight Vector for Optimal Hypothesis: ", adaboost[iter].Dt)

                test_errors /= folds
                train_errors /= folds
                plt.figure()
                plt.scatter(range(0,13), train_errors, marker="x", label="Training error")
                plt.scatter(range(0,13), test_errors, marker="x", label="Test error")
                plt.title("Adaboost with " + str(folds) + " fold Cross Validation")
                plt.xlabel("Number of iterations (T)")
                plt.ylabel("Mean Error")
                plt.legend()
                if (wt_req == "wp"): plt.show()

            else:
                print("For ERM input should be like: "+
                "python3 adaboost.py --dataset /path/to/data/filename.csv --mode erm")
                print("For Cross validation input should be like: " +
                "python3 adaboost.py --dataset /path/to/data/filename.csv --mode cv <folds>")
    except IOError as e:
        print("Couldn't open the file (%s). Check file path and name again." % e)


if __name__ == "__main__":
    main()