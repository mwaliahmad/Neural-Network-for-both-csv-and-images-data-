from strategy import ModelStrategy


class ModelContext:
    def __init__(self, model_strategy: ModelStrategy):
        self._model_strategy = model_strategy

    def load_model(self, model_path: str):
        self._model_strategy.load_model(model_path)

    def predict(self, X, parameters):
        return self._model_strategy.predict(X, parameters)

    def train(self, progress_callback=None):
        return self._model_strategy.train(progress_callback)
