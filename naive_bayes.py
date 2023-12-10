from functools import reduce
import pandas as pd
import pprint

class NaiveBayesClassifier:
    def __init__(self, filename = None, class_attr = None):
        self.data = pd.read_csv(filename, sep=',', header=0)
        self.class_attr = class_attr
        self.priori_probabilities = {}
        self.conditional_probabilities = {}
        self.hypothesis = None

    def calculate_priori_probabilities(self):
        # formula: (Number of occurrences of a class) / (Total number of instances in the dataset)
        
        class_values = list(set(self.data[self.class_attr]))
        class_data = list(self.data[self.class_attr])

        for class_val in class_values:
            self.priori_probabilities[class_val] = class_data.count(class_val) / float(len(class_data))
        print("Priori Probabilities: ", self.priori_probabilities)

    def get_conditional_probability(self, attr, attr_type, class_value):
        # formula: (Number of instances with the given attribute value and class) / (Total number of instances with the given class)
        
        data_attr = list(self.data[attr])
        class_data = list(self.data[self.class_attr])
        total = 0

        for i in range(0, len(data_attr)):
            if class_data[i] == class_value and data_attr[i] == attr_type:
                total += 1

        return total / float(class_data.count(class_value))

    def calculate_conditional_probabilities(self, hypothesis):
        # Calculate and store the conditional probabilities for each attribute given the target class value.

        for class_val in self.priori_probabilities:
            self.conditional_probabilities[class_val] = {}
            for attribute, attr_type in hypothesis.items():
                self.conditional_probabilities[class_val].update({
                    attribute: self.get_conditional_probability(attribute, attr_type, class_val)
                })
        print("\nCalculated Conditional Probabilities: \n")
        pprint.pprint(self.conditional_probabilities)

    def classify(self):
        print("Result: ")
        for class_val in self.conditional_probabilities:
            likelihood = reduce(lambda x, y: x * y, self.conditional_probabilities[class_val].values())
            result = likelihood * self.priori_probabilities[class_val]
            print(class_val, " ==> ", result)