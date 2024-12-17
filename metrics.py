import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

class MatchingMetrics:
    def __init__(self, 
                 label_map, 
                 thresholds):
        
        self.thresholds = thresholds
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def _score_to_label(self, score):
        if score < self.thresholds[0]:
            return 0  # No Fit
        elif score < self.thresholds[1]:
            return 1  # Potential Fit
        else:
            return 2  # Good Fit

    def _value_to_class_name(self, value):
        label_map = {0: 'No Fit', 1: 'Potential Fit', 2: 'Good Fit'}
        return label_map[value]

    def _convert_continuous_to_discrete(self, labels):
        """Convert continuous labels to discrete with floating-point tolerance"""
        discrete_labels = []
        for label in labels:
            if abs(label - self.label_map['No Fit']) < 1e-5:
                discrete_labels.append(0)
            elif abs(label - self.label_map['Potential Fit']) < 1e-5:
                discrete_labels.append(1)
            elif abs(label - self.label_map['Good Fit']) < 1e-5:
                discrete_labels.append(2)
            else:
                raise ValueError(f"Unexpected label value: {label}")
        return np.array(discrete_labels)

    def calculate_metrics(self, scores, true_labels):
        # Convert continuous predictions to discrete labels using thresholds
        predicted_discrete = np.array([self._score_to_label(score) for score in scores])
        
        # Convert continuous true labels to discrete
        true_discrete = self._convert_continuous_to_discrete(true_labels)
        
        # Calculate accuracy with discrete labels
        accuracy = accuracy_score(true_discrete, predicted_discrete)
        
        # Calculate precision, recall, F1 with discrete labels
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_discrete, 
            predicted_discrete,
            labels=[0, 1, 2],  # Discrete labels
            average=None
        )

        f1_weighted = f1_score(true_discrete, predicted_discrete, 
                                labels=[0, 1, 2],
                                average='weighted')
        
        # Calculate confusion matrix with discrete labels
        conf_matrix = confusion_matrix(
            true_discrete,
            predicted_discrete,
            labels=[0, 1, 2]
        )
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(['No Fit', 'Potential Fit', 'Good Fit']):
            class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i]
            }
        
        # Calculate regression metrics using original continuous values
        mse = np.mean((scores - true_labels) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'rmse': rmse,
            'per_class': class_metrics,
            'confusion_matrix': conf_matrix,
            'f1_weighted': f1_weighted
        }

    def print_metrics(self, metrics):
        print("\n=== Classification Metrics ===")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}\n")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        
        print("=== Per-Class Metrics ===")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1']:.4f}")
        
        print("\n=== Confusion Matrix ===")
        print("Rows: True Labels, Columns: Predicted Labels")
        print("Labels order: No Fit (0), Potential Fit (1), Good Fit (2)")
        print(metrics['confusion_matrix'])