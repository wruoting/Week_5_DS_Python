import numpy as np
import pandas as pd
from collections import deque, Counter

class Custom_knn():
    def __init__(self, number_neighbors_k, distance_parameter_p=2):
        super().__init__()
        self.k = number_neighbors_k
        self.p = distance_parameter_p
        self.reciprocal_p = np.divide(1, self.p)
        self.training_values = None
        self.training_labels = None

    def __str__(self):
        return super().__str__()
    
    def fit(self, X, Labels):
        '''
        X: input vector that is an n-dim training selection
        Labels: associated labels that maps 1:1 to each element in X
        returns: None
        '''
        # Assert that X is of the correct dimension and Labels as well
        assert len(X) == len(Labels), "Input vectors need to be the same length as labels"
        self.training_values = X
        self.training_labels = Labels

    def predict(self, new_x):
        '''
        new_x: the new input of what you want to predict
        returns: labels of len new_x with predictions
        '''
        if len(new_x) == 0:
            return None
        list_of_labels = deque()
        for x in new_x:
             list_of_labels.append(self._predict_single(x))
        return np.array(list_of_labels)
       

    def draw_decision_boundary(self, new_x):
        '''
        new_x: the new input of what you want to predict
        returns: k neighbors with ids and colors
        '''
        pass

    def _predict_single(self, x):
        '''
        x: this is a single entry out of an array of inputs
        returns: prediction of single input
        '''
        distance_vector = deque()
        label_vector = deque()
        for point, label in zip(self.training_values, self.training_labels):
            distance_vector.append(self.__calculate_distance(x, point))
            label_vector.append(label)
        # Convert deque to array
        distance_vector_np = np.array(distance_vector)
        label_vector_np = np.array(label_vector)
   
        # Argsort the distance vector and grab the top k indices
        k_closest_indices = np.argsort(distance_vector_np)[0:self.k]
        # Print the labels with the k closest points
        k_closest_labels = [label_vector_np[k] for k in k_closest_indices]

        # Get the counter
        counter_labels = Counter(k_closest_labels)
        # If we don't have any labels
        if len(counter_labels) == 0:
            return None
        # Most common label
        return counter_labels.most_common(1)[0][0]


    def __calculate_distance(self, new_x, point):
        # Calculate vector of absolute difference between point and new_x
        abs_diff = np.abs(np.subtract(new_x, point))
        # Calculate the list of squared absolute diffs
        exponent_abs_point = np.power(abs_diff, self.p)
        # Give a sum of the absolute values
        total_sum = np.sum(exponent_abs_point)
        # The distance is the power of this to the reciprocal
        return np.power(total_sum, self.reciprocal_p)
