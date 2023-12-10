from naive_bayes import NaiveBayesClassifier

model = NaiveBayesClassifier(filename = "new_dataset.csv", class_attr = "Play")
model.calculate_priori_probabilities()
model.hypothesis = {"Outlook": 'Rainy', "Temp": "Mild", "Humidity": 'Normal', "Windy": 't'}
model.calculate_conditional_probabilities(model.hypothesis)
model.classify()