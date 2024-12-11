import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


np.random.seed(42)

df = pd.read_csv('banknote_authentication.csv')

# separation of data
X = df.drop('class', axis=1)
y = df['class']


# splitting of dataset
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
x_valid, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# standardizing features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logisticRegression(x_train, y_train, x_valid, y_val, learningRate, iterations):
    m, n = x_train.shape
    weights = np.zeros(n)
    bias = 0.0
    valSet_accuracies = []

    for _ in range(iterations):
        trainModel = np.dot(x_train, weights) + bias
        y_pred_train = sigmoid(trainModel)

        dw = (1 / m) * np.dot(x_train.T, (y_pred_train - y_train))
        db = (1 / m) * np.sum(y_pred_train - y_train)

        weights -= learningRate * dw
        bias -= learningRate * db

        # storing accuracies for each validation set
        y_pred_val = predict(x_valid, weights, bias)
        valSet_accuracies.append(accuracy(y_val, y_pred_val))

    return weights, bias, max(valSet_accuracies)


def predictedProb(X, weights, bias):
    model = np.dot(X, weights) + bias
    return sigmoid(model)


def predict(X, weights, bias, threshold=0.5):
    probabilities = predictedProb(X, weights, bias)
    return (probabilities >= threshold).astype(int)


# percentage of correctly classified samples
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# proportion of predicted positives that are correct
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


# proportion of actual positives correctly identified
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


# harmonic mean of precision and recall, balances
def f1_score(precision_val, recall_val):
    return 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0


# hyperparameter tuning
learningRates = [0.001, 0.01, 0.1]
iterations_list = [500, 1000]
best_accuracy = 0
best_params = {}
results = []

for lr in learningRates:
    for iters in iterations_list:
        _, _, val_accuracy = logisticRegression(x_train, y_train, x_valid, y_val, learningRate=lr,
                                                       iterations=iters)
        results.append({"Learning Rate": lr, "Iterations": iters, "Validation Accuracy": val_accuracy})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {'learningRate': lr, 'iterations': iters}


results_df = pd.DataFrame(results)
print(results_df)

print(f'\nBest Hyperparameters:\nLearning Rate: {best_params["learningRate"]}, '
      f'Iterations: {best_params["iterations"]}')
print(f'Best Validation Accuracy: {best_accuracy:.2f}')


best_weights, best_bias, _ = logisticRegression(
    x_train, y_train, x_valid, y_val, learningRate=best_params['learningRate'],
    iterations=best_params['iterations']
)


# testing dataset
y_probs = predictedProb(x_test, best_weights, best_bias)
y_pred = predict(x_test, best_weights, best_bias)

test_accuracy = accuracy(y_test, y_pred)
test_precision = precision(y_test, y_pred)
test_recall = recall(y_test, y_pred)
test_f1 = f1_score(test_precision, test_recall)


evaluation_results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [test_accuracy, test_precision, test_recall, test_f1]
})
print("\nTest Set Evaluation Metrics:")
print(evaluation_results)


def rocPlot(y_true, y_probs):
    thresholds = np.linspace(0, 1, num=100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred_binary = (y_probs >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.plot(fpr_list, tpr_list, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


rocPlot(y_test, y_probs)


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
