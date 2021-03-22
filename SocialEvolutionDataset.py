import os.path as osp

import torch
import pandas
import pandas as pd
from torch_geometric.data import InMemoryDataset, TemporalData, download_url


class SocialEvolutionDataset(InMemoryDataset):

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()

        super(SocialEvolutionDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        # df = pandas.read_csv(self.raw_paths[0], skiprows=1, header=None)
        df = pandas.read_csv(self.raw_paths[0])
        df.dt = pd.to_datetime(df.dt)
        df['dt_hr'] = df.dt - df.dt[0]
        def timedelta2hour(x):
            return round((x.days * 24 + x.seconds / 3600), 3)
        df['dt_hr'] = df['dt_hr'].apply(timedelta2hour)
        src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
        t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
        y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
        dt_hr = torch.from_numpy(df.iloc[:, 5].values).to(torch.float)
        # msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)

        data = TemporalData(src=src, dst=dst, t=t, y=y, dt_hr=dt_hr)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name.capitalize()}()'
