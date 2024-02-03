import torch


class Trainer():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
    
    def train_step(self, x, y):
        pass