import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc
)


class SVM_RFF:
    def __init__(self, X_train, X_test, y_train, y_test,
                 learning_rate=0.001, _lambda=0.001, epochs=1000,
                 epsilon=1e-4, gamma=0.01, D=500):
        self.X_train = X_train.values if hasattr(X_train, "values") else X_train
        self.X_test = X_test.values if hasattr(X_test, "values") else X_test
        self.y_train = y_train
        self.y_test = y_test

        self.lr = learning_rate
        self._lambda = _lambda
        self.epochs = epochs
        self.epsilon = epsilon
        self.gamma = gamma
        self.D = D

    def apply_random_fourier_features(self, X):
        projection = X.dot(self.W.T) + self.b
        return np.sqrt(2 / self.D) * np.cos(projection)

    def init_rff(self):
        d = self.X_train.shape[1]
        self.W = np.random.normal(scale=np.sqrt(2 * self.gamma), size=(self.D, d))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.D)

        self.Z_train = self.apply_random_fourier_features(self.X_train)
        self.Z_test = self.apply_random_fourier_features(self.X_test)

    def fit_model(self):
        n_samples, n_features = self.Z_train.shape
        y_ = np.where(self.y_train <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b_svm = 0

        for i in tqdm(range(self.epochs), desc="Training"):
            w_prev = self.w.copy()
            for idx, z_i in enumerate(self.Z_train):
                condition = y_[idx] * (np.dot(z_i, self.w) + self.b_svm) >= 1
                if condition:
                    self.w -= self.lr * (2 * self._lambda * self.w)
                else:
                    self.w -= self.lr * (2 * self._lambda * self.w - y_[idx] * z_i)
                    self.b_svm += self.lr * y_[idx]

            if np.linalg.norm(self.w - w_prev) < self.epsilon:
                print(f"Converged at iteration {i}")
                break

    def decision_function(self, Z):
        return np.dot(Z, self.w) + self.b_svm

    def predict(self, Z):
        return (self.decision_function(Z) >= 0).astype(int)

    def predict_proba(self, Z):
        return self.decision_function(Z)

    def report_metrics(self):
        y_pred = self.predict(self.Z_test)
        accuracy = np.mean(y_pred == self.y_test)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        self.plot_confusion_matrix(self.y_test, y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self):
        y_scores = self.predict_proba(self.Z_test)
        fpr, tpr, _ = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def fit(self):
        self.init_rff()
        self.fit_model()
        self.report_metrics()
