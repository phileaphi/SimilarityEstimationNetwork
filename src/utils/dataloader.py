from torch.utils.data import DataLoader, BatchSampler


class MonkeyLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # num_workers = 0
        # if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
        #     num_workers = kwargs['num_workers']
        #     kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
        super().__setattr__('__finished_init__', True)
        # dataset = self.dataset
        # while hasattr(dataset, 'dataset'):
        #     dataset = dataset.dataset  # This is stupid!

        # self.tranform = self.dataset.transform
        # self.target_tranform = self.dataset.target_tranform

    def __getattr__(self, attr):
        ret = None
        try:
            ret = super(DataLoader, self).__getattribute__(attr)
        except AttributeError:
            super(DataLoader, self).__getattribute__('__finished_init__')
        try:
            ret = super(DataLoader, self).__getattr__(attr)
        except AttributeError:
            pass
        if ret is None:
            raise AttributeError(f'The monkey is sorry because it could not find {attr}')
        else:
            return ret

    def __setattr__(self, attr, val):
        try:
            super(DataLoader, self).__getattribute__(attr)
            super(DataLoader, self).__setattr__(attr, val)
            return
        except AttributeError:
            try:
                super(DataLoader, self).__getattribute__('__finished_init__')
            except AttributeError:
                super(DataLoader, self).__setattr__(attr, val)
                return
        raise AttributeError('The monkey is sorry because it could not set {attr}')

    @property
    def batch_size(self):
        try:
            super(DataLoader, self).__getattribute__('batch_sampler')
            return self.batch_sampler.batch_size
        except AttributeError:
            return self._batch_size

    @batch_size.setter
    def batch_size(self, val):
        try:
            super(DataLoader, self).__getattribute__('batch_sampler')
            self.batch_sampler = BatchSampler(self.sampler, val, self.drop_last)
        except AttributeError:
            self._batch_size = val
