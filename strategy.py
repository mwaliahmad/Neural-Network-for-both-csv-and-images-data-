from abc import ABC, abstractmethod
from utils import save_parameters, load_parameters
from model import Model


class ModelStrategy(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, X, parameters):
        pass

    @abstractmethod
    def train(self):
        pass


class ImageClassificationModel(ModelStrategy):
    def __init__(
        self,
        X,
        Y,
        val_X,
        val_Y,
        layers_config,
        learning_rate,
        num_iterations,
        thread=None,
    ):
        self.model = Model(
            X, Y, val_X, val_Y, layers_config, learning_rate, num_iterations, thread
        )

    def load_model(self, model_path: str):
        self.parameters = load_parameters(model_path)

    def predict(self, X, parameters):
        return self.model.predict(X, parameters)

    def train(self, progress_callback=None):
        self.parameters = self.model.train(progress_callback)
        save_parameters(self.parameters, "Image_Parameters")


class CSVClassificationModel(ModelStrategy):
    def __init__(
        self,
        X,
        Y,
        val_X,
        val_Y,
        layers_config,
        learning_rate,
        num_iterations,
        thread=None,
    ):
        self.model = Model(
            X, Y, val_X, val_Y, layers_config, learning_rate, num_iterations, thread
        )

    def load_model(self, model_path: str):
        self.parameters = load_parameters(model_path)

    def predict(self, X, parameters):
        return self.model.predict(X, parameters)

    def train(self, progress_callback=None):
        self.parameters = self.model.train(progress_callback)
        save_parameters(self.parameters, "CSV_Parameters")
