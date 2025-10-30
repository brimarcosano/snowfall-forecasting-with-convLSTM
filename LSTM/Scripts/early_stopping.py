class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.counter = 0
        
    def early_stop(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss  # Fixed variable name
            self.counter = 0
        elif val_loss > (self.best_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False 
