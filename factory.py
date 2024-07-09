from strategy import ImageClassificationModel, CSVClassificationModel


class ModelFactory:
    @staticmethod
    def get_model(
        model_type: str,
        X,
        Y,
        val_X,
        val_Y,
        layers_config,
        learning_rate,
        num_iterations,
        thread=None,
    ):
        if model_type == "image":
            return ImageClassificationModel(
                X, Y, val_X, val_Y, layers_config, learning_rate, num_iterations, thread
            )
        elif model_type == "csv":
            return CSVClassificationModel(
                X, Y, val_X, val_Y, layers_config, learning_rate, num_iterations, thread
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
