import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Annotated, Tuple, TypeVar, Protocol

"""
    - Torch isnt supported for Python 3.14
    - Heres what I envision this would look like (beware, it has errors!)
    
    # Python 3.13
    type Tensor = numpy.ndarray | torch.Tensor
    type Tensors = Tuple[Tensor, ...]
    type TensorMatch[T: Tensor, LEN: (LengthContraint)] = Annotated[T, LEN]

    ## If you use different tensor types this would raise an error
    class Data[T: Tensor, LEN: (LengthContraint)](TypedDict, total=True):
        _type: str
        test: TensorMatch
        train: TensorMatch
        
    # OR in Python 3.14
    ## Data[T: Tensor, LEN: (LengthContraint)] = TypedDict[{'type': str, 'test': TensorMatch, 'train': TensorMatch}]
"""

# Define a protocol to ensure objects have a __len__ method
class HasLength(Protocol):
    def __len__(self) -> int: ...

# Define a generic type variable constrained to types with __len__
T = TypeVar('T', bound=HasLength)

# Length constraint metadata class
class LengthConstraint:
    def __init__(self, min_length: int = 0, max_length: int = float('inf')):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: HasLength):
        if not (self.min_length <= len(value) <= self.max_length):
            raise ValueError(f'Value length must be between {self.min_length} and {self.max_length}, got {len(value)}')

# Define a generic length-bounded annotation
TensorMatch = Annotated[T, LengthConstraint]

# Custom dictionary class to allow dot notation access
class Data(dict):
    def __init__(self, _type: str, test: TensorMatch, train: TensorMatch):
        super().__init__(_type=_type, test=test, train=train)
        self._type = _type
        self.test = test
        self.train = train

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Data' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value
        super().__setattr__(key, value)

# Function to load data
def load_data(option: str = 'mnist_784', split: int = 60000, scaler: int = 255, convert: bool = True) -> Tuple[Data, Data]:
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(option, version=1, return_X_y=True, parser='auto')

    x_data = Data('x tensors', X[split:], X[:split])
    y_data = Data('y tensors', y[split:], y[:split])

    if convert:
        x_data.test = np.array(x_data.test) / scaler
        x_data.train = np.array(x_data.train) / scaler
        y_data.test = np.array(y_data.test).astype(int)
        y_data.train = np.array(y_data.train).astype(int)

    return x_data, y_data

# Function to display images
def show_images(tensors: Tuple[Data, Data], cnt: int = 3):
    plt.figure(figsize=(20, 4))

    for index, (image, label) in enumerate(zip(tensors[0].train[:cnt], tensors[1].train[:cnt])):
        plt.subplot(1, cnt, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
        plt.title(f'Label: {label}', fontsize=20)
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier

    # Load data
    tensors = load_data()

    # Create an MLPClassifier object
    mlp = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=10,
        alpha=1e-4,
        solver='sgd',
        verbose=10,
        random_state=1,
        learning_rate_init=0.1,
    )

    # Train the MLPClassifier
    mlp.fit(tensors[0].train, tensors[1].train)

    # Show the accuracy on the training and test sets
    print(f'Training set score: {mlp.score(tensors[0].train, tensors[1].train)}')
    print(f'Test set score: {mlp.score(tensors[0].test, tensors[1].test)}')

    # Show the images, predictions, and original labels for 10 images
    predictions = mlp.predict(tensors[0].test[:10])

    plt.figure(figsize=(8, 4))

    for index, (image, prediction, label) in enumerate(zip(tensors[0].test[:10], predictions, tensors[1].test[:10])):
        plt.subplot(2, 5, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
        fontcolor = 'g' if prediction == label else 'r'
        plt.title(f'Prediction: {prediction}\nLabel: {label}', fontsize=10, color=fontcolor)
        plt.axis('off')

    plt.show()

