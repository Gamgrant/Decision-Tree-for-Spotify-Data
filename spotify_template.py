"""
Code to build decision tree classifier for spotify data
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def load_data():
    """ Load Data Set, making sure to import the index column correctly
        Arguments:
            None
        Returns:
            Training data dataframe, training labels, testing data dataframe,
            testing labels, features list
    """
    # TODO: Finish this function



def cv_grid_search(training_table, training_labels):
    """ Run grid search with cross-validation to try different
    hyperparameters
        Arguments:
            Training data dataframe and training labels
        Returns:
            Dictionary of best hyperparameters found by a grid search with
            cross-validation
    """
    # TODO: Finish this function


def plot_confusion_matrix(test_labels, pred_labels):
    """Plot confusion matrix
        Arguments:
            ground truth labels and predicted labels
        Returns:
            Writes image file of confusion matrix
    """
    # TODO: Finish this function



def graph_tree(model, training_features, class_names):
    """ Plot the tree of the trained model
        Arguments:
            Trained model, list of features, class names
        Returns:
            Writes PDF file showing decision tree representation
    """
    # TODO: Finish this function


def print_results(predictions, test_y):
    """Print results
        Arguments:
            Ground truth labels and predicted labels
        Returns:
            Prints precision, recall, F1-score, and accuracy
    """
    # TODO: Finish this function


def print_feature_importance(model, features):
    """Print feature importance
        Arguments:
            Trained model and list of features
        Returns:
            Prints ordered list of features, starting with most important,
            along with their relative importance (percentage).
    """
    # TODO: Finish this function


def main():
    """Run the program"""
    # Load data
    (train_x, test_x, train_y, test_y), features = load_data()

    # Cross Validation Training
    params = cv_grid_search(train_x, train_y)
    # params = ['entropy', 4, 'balanced']

    # Train and test model using hyperparameters
    # TODO: Finish this function

    # Confusion Matrix
    plot_confusion_matrix(test_y, list(predictions))

    # Graph Tree
    graph_tree(model, features, ['hate', 'love'])

    # Accuracy, Precision, Recall, F1
    print_results(predictions, test_y)

    # Feature Importance
    print_feature_importance(model, features)


if __name__ == '__main__':
    main()
