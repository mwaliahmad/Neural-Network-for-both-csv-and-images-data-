import json
import os


class ConfigLoader:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as config_file:
            return json.load(config_file)

    def get_image_config(self):
        return self.config["image_classification"]

    def get_csv_config(self):
        return self.config["csv_classification"]

    def get_data_paths(self):
        return self.config["data_paths"]

    def update_config(self, key, value):
        if key in self.config:
            self.config[key] = value
            self.save_config()
        else:
            raise KeyError(f"Configuration key not found: {key}")

    def save_config(self):
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file, indent=4)

    def get_model_config(self, model_type):
        if model_type == "image":
            return self.get_image_config()
        elif model_type == "csv":
            return self.get_csv_config()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    config = ConfigLoader()

    image_config = config.get_image_config()
    print("Image Classification Config:", image_config)

    csv_config = config.get_csv_config()
    print("CSV Classification Config:", csv_config)

    data_paths = config.get_data_paths()
    print("Data Paths:", data_paths)

    config.update_config(
        "data_paths",
        {
            "image_dataset": "new/path/to/image/dataset",
            "csv_dataset": "new/path/to/csv/dataset.csv",
        },
    )

    updated_data_paths = config.get_data_paths()
    print("Updated Data Paths:", updated_data_paths)
