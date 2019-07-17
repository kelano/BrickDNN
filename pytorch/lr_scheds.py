class LRScheduler:
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.lr = init_lr
        self.done = False

    def update_learning_rate(self, dev_losses):
        return

    def is_done(self):
        return self.done

    def _set_new_lr(self, new_lr):
        print('Changing learning rate {} ==> {}'.format(self.lr, new_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr = new_lr


class XValPlateauLRScheduler(LRScheduler):
    def __init__(self, optimizer, init_lr, plateau_history_size, plateaus_to_decay, max_decays, decay_factor):
        LRScheduler.__init__(self, optimizer, init_lr)
        self.plateau_history_size = plateau_history_size
        self.plateaus_to_decay = plateaus_to_decay
        self.max_decays = max_decays
        self.decay_factor = decay_factor
        self.lr_decays = 0
        self.plateaus = []

    def update_learning_rate(self, dev_losses):
        # decay LR if necessary
        if len(dev_losses) >= 2:
            self.plateaus.append(1 if (dev_losses[-1] > dev_losses[-2]) else 0)
        else:
            self.plateaus.append(0)

        if len(self.plateaus) >= self.plateau_history_size and sum(self.plateaus[-self.plateau_history_size:]) >= self.plateaus_to_decay:
            del self.plateaus[:]
            self.lr_decays += 1
            if self.lr_decays >= self.max_decays:
                self.done = True
                return
            new_lr = self.lr * self.decay_factor
            self._set_new_lr(new_lr)


class LRTestScheduler(LRScheduler):
    def __init__(self, optimizer, lr_min=0.0, lr_max=0.2, lr_step=0.01):
        LRScheduler.__init__(self, optimizer, lr_min)
        self.lr_max = lr_max
        self.lr_step = lr_step

    def update_learning_rate(self, dev_losses):
        new_lr = self.lr + self.lr_step
        if new_lr > self.lr_max:
            self.done = True
            return
        self._set_new_lr(new_lr)
