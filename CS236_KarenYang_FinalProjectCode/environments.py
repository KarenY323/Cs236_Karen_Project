import os
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import pickle as pkl
import numpy as np

from data.base_dataset import BaseDataset, get_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_groups = ['trainA', 'trainB']

class FIGR8MetaEnv(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task()

    def get_tasks(self):
        if os.path.exists('./data/FIGR-8') is False:
            if os.path.exists('./data') is False:
                os.mkdir('./data')
            os.mkdir('./data/FIGR-8')
            from google_drive_downloader import GoogleDriveDownloader as gdd
            gdd.download_file_from_google_drive(file_id='10dF30Qqi9RdIUmET9fBhyeRN0hJmq7pO',
                                                dest_path='./data/FIGR-8/Data.zip')
            import zipfile
            with zipfile.ZipFile('./data/FIGR-8/Data.zip', 'r') as zip_f:
                zip_f.extractall('./data/FIGR-8/')
            os.remove('./data/FIGR-8/Data.zip')


        tasks = dict()
        path = './data/FIGR-8/Data'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = os.path.join(path, task)
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.to_tensor(self.resize(img))))
            tasks[task] = np.array(tasks[task])
        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 50))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def __len__(self):
        return len(self.files)


class KarenMetaEnv(Dataset):
    def __init__(self, height=256, length=256, num_images=25):
        self.channels = 3
        self.height = height
        self.length = length
        self.num_images = num_images
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task()

    def get_tasks(self):
        # if os.path.isfile('./tasks.pkl') is True:
        #     print('Using `tasks.pkl` for self.tasks')
        #     return pkl.load(open('tasks.pkl', 'wb'))

        if os.path.exists('./data/Karen') is False:
            if os.path.exists('./data') is False:
                os.mkdir('./data')
            os.mkdir('./data/Karen')
            raise("Please download Karen data and move it to ./data/Karen")
            # from google_drive_downloader import GoogleDriveDownloader as gdd
            # gdd.download_file_from_google_drive(file_id='10dF30Qqi9RdIUmET9fBhyeRN0hJmq7pO',
            #                                     dest_path='./data/FIGR-8/Data.zip')
            # import zipfile
            # with zipfile.ZipFile('./data/Karen/Data.zip', 'r') as zip_f:
            #     zip_f.extractall('./data/Karen/')
            # os.remove('./data/Karen/Data.zip')

        tasks = dict()
        path = './data/Karen/Data'
        task_names = [ task_name for task_name in os.listdir(path) if task_name != '.DS_Store' ]
        print(task_names)

        # for task in task_names:
        for task in ['train_vangogh', 'train_apple', 'train_yosemite_summer', 'train_yosemite_winter']:
            task_path = os.path.join(path, task)
            print(task_path)

            tensor_images = [
                np.array(self.to_tensor(self.resize(Image.open(os.path.join(task_path, img_name)))))
                for img_name in os.listdir(task_path)[:self.num_images]
            ]

            tasks[task] = np.array(tensor_images)

        # pkl.dump(tasks, open('tasks.pkl', 'wb'))
        return tasks

    def split_validation_and_training_task(self):
        # self.validation_task = set(random.sample(self.all_tasks, 50))
        self.validation_task = {'train_yosemite_winter'}

        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample(list(range(self.tasks[task].shape[0])), batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample(list(range(self.tasks[task].shape[0])), batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def __len__(self):
        return len(self.files)


class KarenMetaEnv2(Dataset):
    def __init__(self, height=256, length=256, num_images=25):
        self.channels = 3
        self.height = height
        self.length = length
        self.num_images = num_images
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.all_tasks = set(self.tasks)
        self.split_validation_and_training_task()

    def get_tasks(self):
        tasks = dict()
        path = './datasets'
        task_names = [
            task_name for task_name in os.listdir(path)
            if (task_name != '.DS_Store')
                and ('.py' not in task_name)
                and ('.sh' not in task_name)
        ]
        print(task_names)

        # for task in task_names:
        for task in ['apple2orange', 'cezanne2photo', 'summer2winter', 'horse2zebra']:
            tasks[task] = dict()
            for group in train_groups:
                task_path = os.path.join(path, task, group)
                print(task_path)

                tensor_images = [
                    np.array(self.to_tensor(self.resize(Image.open(os.path.join(task_path, img_name)))))
                    for img_name in os.listdir(task_path)[:self.num_images]
                ]
                #[(3, 32, 32), ...]

                image_names = [
                    os.path.join(task_path, img_name)
                    for img_name in os.listdir(task_path)[:self.num_images]
                ]

                tasks[task][group] = [ (tensor, name) for tensor, name in zip(tensor_images, image_names) ]

                '''


                    input_nc = self.opt.output_nc if btoA else self.opt.input_nc
                    output_nc = self.opt.input_nc if btoA else self.opt.output_nc
                    self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
                    self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

                    A_img = Image.open(A_path).convert('RGB')
                    B_img = Image.open(B_path).convert('RGB')

                    A = self.transform_A(A_img)
                    B = self.transform_B(B_img)
                '''

        return tasks

    def split_validation_and_training_task(self):
        # self.validation_task = set(random.sample(self.all_tasks, 50))
        self.validation_task = {'horse2zebra'}

        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]

        batch_list = []
        for group in train_groups:
            task_idx = random.sample(list(range(len(self.tasks[task][group]))), batch_size)
            batch = [ self.tasks[task][group][i] for i in task_idx ]
            # batch = [
            #     (
            #         torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0),
            #         name,
            #         torch.tensor(img, dtype=torch.float, device=device),
            #     )
            #     for img, name in batch
            # ]
            batch = [ (torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0), name) for img, name in batch ]
            batch_list.append(batch)

        return batch_list[0], batch_list[1], task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]

        batch_list = []
        for group in train_groups:
            task_idx = random.sample(list(range(len(self.tasks[task][group]))), batch_size)
            batch = [ self.tasks[task][group][i] for i in task_idx ]
            # batch = [
            #     (
            #         torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0),
            #         name,
            #         torch.tensor(img, dtype=torch.float, device=device),
            #     )
            #     for img, name in batch
            # ]
            batch = [ (torch.tensor(img, dtype=torch.float, device=device).unsqueeze(0), name) for img, name in batch ]
            batch_list.append(batch)

        return batch_list[0], batch_list[1], task

    def __len__(self):
        return len(self.files)
