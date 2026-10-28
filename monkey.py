import random
from nvidiadata import dataset
correct = 0

for inputs, down, neutral, up in dataset:

    actual = [down, neutral, up].index(1)

    prediction = random.randint(0, 2)

    if prediction == actual:
        correct += 1

accuracy = (correct / len(dataset)) * 100

print("Random Accuracy:", round(accuracy, 2), "%")