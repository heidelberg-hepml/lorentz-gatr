import torch


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, events, dtype):
        self.events = [
            torch.tensor(events_onedataset, dtype=dtype) for events_onedataset in events
        ]
        self.lens = [len(events_onedataset) for events_onedataset in self.events]

    def __len__(self):
        return max(self.lens)

    def __getitem__(self, idx):
        # if sub-dataset has less than max(self.lens) events,
        # some events will be sampled more than one time
        # Note that the model sees events with smaller idx more often
        return [events[idx % _len] for events, _len in zip(self.events_eff, self.lens)]


class EventDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # manually shuffle the dataset after each epoch
            # this is necessary to avoid having small-idx events more often in the custom __getitem__ method
            perms = [torch.randperm(len(events)) for events in self.dataset.events]
            self.dataset.events_eff = [
                events[perm] for events, perm in zip(self.dataset.events, perms)
            ]

        return super().__iter__()
