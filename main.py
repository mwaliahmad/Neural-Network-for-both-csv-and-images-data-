import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from PIL import Image
import pandas as pd
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QTextEdit,
    QLabel,
    QProgressBar,
)
from utils import load_parameters
from thread import Thread
from factory import ModelFactory
from context import ModelContext
from config_loader import ConfigLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = ConfigLoader()
        self.images_parameters = load_parameters("Image_Parameters")
        self.csv_parameters = load_parameters("CSV_Parameters")
        self.setWindowTitle("Neural Network Trainer")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()

        image_layout = QVBoxLayout()
        image_layout.addWidget(QLabel("Image Classification"))
        self.image_data_btn = QPushButton("Select Image Dataset")
        self.image_data_btn.clicked.connect(lambda: self.select_data("image"))
        image_layout.addWidget(self.image_data_btn)
        self.image_train_btn = QPushButton("Train Image Model")
        self.image_train_btn.clicked.connect(lambda: self.train_model("image"))
        image_layout.addWidget(self.image_train_btn)
        self.image_predict_btn = QPushButton("Predict Image")
        self.image_predict_btn.clicked.connect(lambda: self.predict("image"))
        image_layout.addWidget(self.image_predict_btn)
        self.image_progress = QProgressBar()
        image_layout.addWidget(self.image_progress)

        csv_layout = QVBoxLayout()
        csv_layout.addWidget(QLabel("CSV Classification"))
        self.csv_data_btn = QPushButton("Select CSV Dataset")
        self.csv_data_btn.clicked.connect(lambda: self.select_data("csv"))
        csv_layout.addWidget(self.csv_data_btn)
        self.csv_train_btn = QPushButton("Train CSV Model")
        self.csv_train_btn.clicked.connect(lambda: self.train_model("csv"))
        csv_layout.addWidget(self.csv_train_btn)
        self.csv_predict_btn = QPushButton("Predict CSV")
        self.csv_predict_btn.clicked.connect(lambda: self.predict("csv"))
        csv_layout.addWidget(self.csv_predict_btn)
        self.csv_progress = QProgressBar()
        csv_layout.addWidget(self.csv_progress)

        # Combine layouts
        data_layout = QHBoxLayout()
        data_layout.addLayout(image_layout)
        data_layout.addLayout(csv_layout)
        main_layout.addLayout(data_layout)

        # Output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

        # Plotting area
        self.plot_figure, self.plot_ax = plt.subplots()
        self.plot_canvas = FigureCanvasQTAgg(self.plot_figure)
        main_layout.addWidget(self.plot_canvas)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize model contexts
        self.image_context = None
        self.csv_context = None

    def select_data(self, data_type):
        if data_type == "image":
            folder = QFileDialog.getExistingDirectory(
                self, "Select Image Dataset Folder"
            )
            if folder:
                self.image_data_path = folder
                self.log_output(f"Selected image dataset: {folder}")
                return folder
        elif data_type == "csv":
            file, _ = QFileDialog.getOpenFileName(
                self, "Select CSV Dataset", "", "CSV Files (*.csv)"
            )
            if file:
                self.csv_data_path = file
                self.log_output(f"Selected CSV dataset: {file}")
                return file

    def train_model(self, model_type):
        model_config = self.config.get_model_config(model_type)
        layers = [
            (layer["units"], layer["activation"]) for layer in model_config["layers"]
        ]

        if model_type == "image":
            if not hasattr(self, "image_data_path"):
                self.log_output("Please select an image dataset first.")
                return
            X, Y = self.load_image_data(self.image_data_path)
            train_X, val_X, train_Y, val_Y = train_test_split(
                X.T, Y.T, test_size=0.2, random_state=1
            )

        elif model_type == "csv":
            if not hasattr(self, "csv_data_path"):
                self.log_output("Please select a CSV dataset first.")
                return
            X, Y = self.load_csv_data(self.csv_data_path)
            train_X, val_X, train_Y, val_Y = train_test_split(
                X.T, Y.T, test_size=0.2, random_state=1
            )

        train_thread = Thread(model_type)
        model = ModelFactory.get_model(
            model_type,
            train_X.T,
            train_Y.T,
            val_X.T,
            val_Y.T,
            layers,
            model_config["learning_rate"],
            model_config["num_iterations"],
            train_thread,
        )

        context = ModelContext(model)
        train_thread.context = context
        train_thread.update_progress.connect(
            self.update_image_progress
            if model_type == "image"
            else self.update_csv_progress
        )
        train_thread.update_plot.connect(self.update_plot)
        train_thread.training_complete.connect(
            lambda: self.training_completed(model_type)
        )
        train_thread.start()

        setattr(self, f"{model_type}_context", context)
        setattr(self, f"{model_type}_train_thread", train_thread)

    def predict(self, model_type):
        if model_type == "image":
            if self.image_context is None:
                self.log_output("Loading image model parameters...")
                self.images_parameters = load_parameters("Image_Parameters")
                model_config = self.config.get_model_config(model_type)
                layers = [
                    (layer["units"], layer["activation"])
                    for layer in model_config["layers"]
                ]
                model = ModelFactory.get_model(
                    model_type,
                    None,
                    None,
                    None,
                    None,
                    layers,
                    model_config["learning_rate"],
                    model_config["num_iterations"],
                )
                self.image_context = ModelContext(model)

                self.log_output("Image model parameters loaded.")

            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image for Prediction",
                "",
                "Image Files (*.png *.jpg *.bmp)",
            )
            if file:
                sample_image = Image.open(file).resize((64, 64))
                sample_image_array = (
                    np.array(sample_image).flatten().reshape(-1, 1) / 255.0
                )
                prediction = self.image_context.predict(
                    sample_image_array, self.images_parameters
                )
                output = "Cat" if prediction == 0 else "Dog"
                self.log_output(f"Image Classification Prediction: {output}")

        elif model_type == "csv":
            if self.csv_context is None:
                self.log_output("Loading CSV model parameters...")
                self.csv_parameters = load_parameters("CSV_Parameters")
                model_config = self.config.get_model_config(model_type)
                layers = [
                    (layer["units"], layer["activation"])
                    for layer in model_config["layers"]
                ]
                model = ModelFactory.get_model(
                    model_type,
                    None,
                    None,
                    None,
                    None,
                    layers,
                    model_config["learning_rate"],
                    model_config["num_iterations"],
                )
                self.csv_context = ModelContext(model)

                self.log_output("CSV model parameters loaded.")

            file, _ = QFileDialog.getOpenFileName(
                self, "Select CSV File for Prediction", "", "CSV Files (*.csv)"
            )
            if file:
                sample_data = pd.read_csv(file)
                prediction = self.csv_context.predict(
                    sample_data.values.T, self.csv_parameters
                )
                output = "Diabetic" if prediction == 1 else "Not Diabetic"
                self.log_output(f"CSV Classification Prediction: {output}")

    def load_image_data(self, data_path):
        X = []
        y = []
        for label in ["1", "0"]:
            folder_path = os.path.join(data_path, label)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(folder_path, filename)
                    img = Image.open(img_path).resize((64, 64))
                    img_array = np.array(img) / 255.0
                    X.append(img_array.flatten())
                    y.append(1 if label == "1" else 0)
        X = np.array(X).reshape(len(X), -1).T
        y = np.array(y).reshape(1, len(y))
        return X, y

    def load_csv_data(self, data_path):
        data = pd.read_csv(data_path)
        y = data.iloc[:, -1].values
        data = data.iloc[:, :-1]
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(0)
        X = data.values
        X = X.T
        y = y.reshape(1, y.shape[0])
        return X, y

    def log_output(self, message):
        self.output_text.append(message)

    def update_image_progress(self, iteration, cost):
        self.image_progress.setValue(
            int(
                (iteration / self.config.get_model_config("image")["num_iterations"])
                * 100
            )
        )
        self.log_output(f"Iteration {iteration}, Cost: {cost}")

    def update_csv_progress(self, iteration, cost):
        self.csv_progress.setValue(
            int(
                (iteration / self.config.get_model_config("csv")["num_iterations"])
                * 100
            )
        )
        self.log_output(f"Iteration {iteration}, Cost: {cost}")

    def update_plot(self, train_costs, val_costs):
        self.plot_ax.clear()
        self.plot_ax.plot(np.squeeze(train_costs), label="Training Cost")
        self.plot_ax.plot(np.squeeze(val_costs), label="Validation Cost")
        self.plot_ax.set_ylabel("Cost")
        self.plot_ax.set_xlabel("Iterations (per hundreds)")
        self.plot_ax.set_title(
            "Learning rate = "
            + str(self.config.get_model_config("image")["learning_rate"])
        )
        self.plot_ax.legend()
        self.plot_canvas.draw()

    def training_completed(self, model_type):
        self.log_output(f"{model_type.capitalize()} model training completed.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
