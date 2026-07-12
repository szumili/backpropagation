import numpy as np

from sklearn.metrics import accuracy_score, classification_report 
from tqdm import tqdm



def metrics(train_set, test_set, neural_network):

    y_label_train = np.argmax(train_set['y'], axis=1)
    y_label_test = np.argmax(test_set['y'], axis=1)

    y_pred_train = [neural_network.predict(digit) for digit in tqdm(train_set['x'], desc="Predicting y_train")]
    y_pred_test = [neural_network.predict(digit) for digit in tqdm(test_set['x'], desc="Predicting y_test")]

    y_pred_label_train = np.argmax(y_pred_train, axis=1)
    y_pred_label_test = np.argmax(y_pred_test, axis=1)

    accuracy_train = accuracy_score(y_label_train, y_pred_label_train)
    accuracy_test = accuracy_score(y_label_test, y_pred_label_test)

    cr_train = classification_report(y_label_train, y_pred_label_train, output_dict=True)
    cr_test = classification_report(y_label_test, y_pred_label_test, output_dict=True)
    
    return accuracy_train, accuracy_test, cr_train, cr_test

