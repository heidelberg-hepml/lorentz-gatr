import torch


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, events, dtype):
        self.events = [
            torch.tensor(events_onedataset, dtype=dtype)
            for events_onedataset in events
        ]

        # reduce the effectively used dataset to the length of the smallest dataset
        # (pure convenience, could use more data at the cost of more code)
        self.len = min(
            [len(events_onedataset) for events_onedataset in self.events]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [events[idx] for events in self.events]
