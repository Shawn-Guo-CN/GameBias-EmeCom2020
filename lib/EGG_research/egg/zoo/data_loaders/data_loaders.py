import torch
import numpy as np
import torch.utils.data
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None, image=False, one_hot=False):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data, encoding='bytes')['imgs'][::subsample]
        self.labels = np.load(path_to_data, encoding='bytes')['latents_classes'][::subsample]
        self.latent_values = np.load(path_to_data, encoding='bytes')['latents_values'][::subsample]

        self.transform = transform
        self.image = image
        self.one_hot = one_hot

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values

        label = self.labels[idx]
        image = self.imgs[idx]*255
        latent_values = self.latent_values[idx] # latents_values: (737280 x 6, float64) Values of the latent factors.

        if self.image:
            image = image.reshape(image.shape + (1,))
            if self.transform:
                image = self.transform(image)

            # drop color - not a factor of variation
            latent_values = np.array(latent_values[1:])
            label = np.array(label[1:])

            # one hotify the categorical variable (shape)
            b = np.zeros(3)
            b[int(latent_values[0] - 1)] = 1

            latent_values = np.concatenate((b, latent_values[1:]), axis=0)

            return image.float(), torch.from_numpy(latent_values).unsqueeze(1).float(), torch.from_numpy(label).unsqueeze(1).float()

        else: # compo vs gen game
            if self.one_hot:
                # TODO
                return np.array(latent_values, dtype='float32'), label
            else:
                return np.array(latent_values, dtype='float32'), label

def get_dsprites_dataloader(batch_size=32,
                            validation_split=.1,
                            random_seed=42,
                            shuffle=True,
                            path_to_data='dsprites.npz',
                            subsample=1,
                            image=False):
    """DSprites dataloader."""

    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor(),
                                    subsample=subsample,
                                    image=image)

    dataset_size = len(dsprites_data)

    print("Dataset size: ", dataset_size)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dsprites_data, batch_size=batch_size,
                                               sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dsprites_data, batch_size=batch_size,
                                                    sampler=valid_sampler, drop_last=True)
    return train_loader, validation_loader

def get_dsprites(batch_size=32, num_workers=1, subsample=1000, validation_split=0.1, shuffle=True, random_seed=7):
<<<<<<< HEAD
    root = os.path.join('dsprites.npz')
    if not os.path.exists(root):
        import subprocess
        print('Now download dsprites-dataset')
        subprocess.call(['./download_dsprites.sh'])
        print('Finished')

    data = np.load(root, encoding='bytes')
    data = torch.from_numpy(data['imgs'][::subsample]).unsqueeze(1).float()
=======




>>>>>>> bb004d266a1e366a60d38b998fc8664c1d05e53b

    dataset_size = len(data)

    print("The size of the entire dataset: ", dataset_size)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers,
                                               pin_memory=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers,
                                                    pin_memory=True, drop_last=True)
    return train_loader, validation_loader
