import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
import math


def main():
    # Load training data
    print 'Loading training data.'
    W_train, W_valid, X_train, X_valid, Y_train, Y_valid = load_training_data()

    # Train the GradientBoostingClassifier using our good features
    print 'Training classifier (this may take some time!)'

    #- AMS based on 90% training   sample: 3.44652047468
    #- AMS based on 10% validation sample: 3.41162329821
    #gbc = GBC(n_estimators=100, max_depth=5,min_samples_leaf=200,max_features=20,verbose=1)

    gbc = svm.LinearSVC(verbose=1)

    #gbc = SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)

    #- AMS based on 90% training   sample: 0.98786705059
    #- AMS based on 10% validation sample: 0.96641705142
    #gbc = GaussianNB()

    #- AMS based on 90% training   sample: 2.88117956464
    #- AMS based on 10% validation sample: 2.98217585696
    #gbc = AdaBoostClassifier(base_estimator=tree.ExtraTreeClassifier(), n_estimators=500)

    #- AMS based on 90% training   sample: 0.0
    #- AMS based on 10% validation sample: 0.0
    #gbc = tree.DecisionTreeClassifier()

    gbc.fit(X_train,Y_train)

    print 'Data Classes --- ', gbc.classes_

    # Get the probaility output from the trained method, using the 10% for testing
    #prob_predict_train = gbc.decision_function(X_train)
    #prob_predict_valid = gbc.decision_function(X_valid)

    # This are the final signal and background predictions
    Yhat_train = gbc.predict(X_train)
    Yhat_valid = gbc.predict(X_valid)

    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.9)
    TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.9)
    TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
    TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)

    # s and b for the training
    s_train = sum ( TruePositive_train*Yhat_train )
    b_train = sum ( TrueNegative_train*Yhat_train )
    s_valid = sum ( TruePositive_valid*Yhat_valid )
    b_valid = sum ( TrueNegative_valid*Yhat_valid )

    # Now calculate the AMS scores

    print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)
    print '   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid)

    # Now we load the testing data, storing the data (X) and index (I)
    print 'Loading testing data'
    data_test = np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )
    X_test = data_test[:,1:31]
    I_test = list(data_test[:,0])

    # Get a vector of the probability predictions which will be used for the ranking
    print 'Building predictions'
    Predictions_test = gbc.predict_proba(X_test)[:,1]
    # Assign labels based the best pcut
    Label_test = list(Predictions_test>pcut)
    Predictions_test =list(Predictions_test)

    # Now we get the CSV data, using the probability prediction in place of the ranking
    print 'Organizing the prediction results'
    resultlist = []
    for x in range(len(I_test)):
        resultlist.append([int(I_test[x]), Predictions_test[x], 's'*(Label_test[x]==1.0)+'b'*(Label_test[x]==0.0)])

    # Sort the result list by the probability prediction
    resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1])

    # Loop over result list and replace probability prediction with integer ranking
    for y in range(len(resultlist)):
        resultlist[y][1]=y+1

    # Re-sort the result list according to the index
    resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])

    # Write the result list data to a csv file
    print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
    fcsv = open('Kaggle_higgs_prediction_output_svm.csv','w')
    fcsv.write('EventId,RankOrder,Class\n')
    for line in resultlist:
        theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
        fcsv.write(theline)
    fcsv.close()

def load_training_data():
    data_train = np.loadtxt('training.csv', delimiter=',', skiprows=1,
                            converters={32: lambda x: int(x == 's'.encode('utf-8'))})
    # Pick a random seed for reproducible results. Choose wisely!
    np.random.seed(42)
    # Random number for training/validation splitting
    r = np.random.rand(data_train.shape[0])
    # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
    print 'Assigning data to numpy arrays.'
    # First 90% are training
    Y_train = data_train[:, 32][r < 0.9]  # label for the row (select ones which are )
    X_train = data_train[:, 1:31][r < 0.9]  # data
    W_train = data_train[:, 31][r < 0.9]  # weight for the row
    # Last 10% are validation
    Y_valid = data_train[:, 32][r >= 0.9]
    X_valid = data_train[:, 1:31][r >= 0.9]
    W_valid = data_train[:, 31][r >= 0.9]
    return W_train, W_valid, X_train, X_valid, Y_train, Y_valid

def AMSScore(s,b):
    return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))

if __name__ == '__main__':
    main()
