import torch


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, events, dtype):
        self.events = [
            torch.tensor(events_onedataset, dtype=dtype) for events_onedataset in events
        ]

        # The model should see each event class (eg ttbar+0j, ttbar+1j etc) the same number of times
        # We implement this by defining epochs as 'the model sees the same amount of samples from
        # each event class', and the minimum-event class determines how long an epoch lasts'
        # We re-create the 'used' part of the dataset for each epoch, see the EventDataLoader class
        self.len = min([len(events_onedataset) for events_onedataset in self.events])
        self.events_eff = [events[: self.len] for events in self.events]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [events[idx] for events in self.events_eff]


class EventDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        # re-sample the used data in each epoch
        perms = [
            torch.randperm(len(events))[: self.dataset.len]
            for events in self.dataset.events
        ]
        self.dataset.events_eff = [
            events[perm] for events, perm in zip(self.dataset.events, perms)
        ]

        return super().__iter__()
