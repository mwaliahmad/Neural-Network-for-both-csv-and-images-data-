from PyQt5.QtCore import QThread, pyqtSignal


class Thread(QThread):
    update_progress = pyqtSignal(int, float)
    training_complete = pyqtSignal()
    update_plot = pyqtSignal(list,list)

    def __init__(self, model_type):
        super().__init__()
        self.context = None
        self.model_type = model_type

    def run(self):
        if self.context is None:
            raise ValueError("Model context is not set")
        self.context.train(self.progress_callback)
        self.training_complete.emit()

    def progress_callback(self, iteration, cost, train_costs, val_costs):
        self.update_progress.emit(iteration, cost)
        self.update_plot.emit(train_costs, val_costs)
