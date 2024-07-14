import os
from PIL import Image
from skimage import io, filters
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import warnings
warnings.filterwarnings("ignore")
from unet_model import UNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import logging
logging.basicConfig(level=logging.INFO, filename='training.log')

# Step 2: Downscale the images to (448,256) (Set A)
def downscale_images(sharp_images_dir, downscaled_dir):
    if not os.path.exists(downscaled_dir):
        os.makedirs(downscaled_dir)
    for subdir in os.listdir(sharp_images_dir):
        subdir_path = os.path.join(sharp_images_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = Image.open(img_path)
                img_resized = img.resize((448, 256))
                save_path = os.path.join(downscaled_dir, subdir, filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img_resized.save(save_path)

# Step 3: Create a set of images by applying different Gaussian filters (Set B)
def apply_gaussian_filters(sharp_images_dir, gaussian_dir):
    if not os.path.exists(gaussian_dir):
        os.makedirs(gaussian_dir)
    kernel_sizes = [3, 7, 11]
    sigmas = [0.3, 1, 1.6]
    for subdir in os.listdir(sharp_images_dir):
        subdir_path = os.path.join(sharp_images_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = io.imread(img_path)
                for kernel_size, sigma in zip(kernel_sizes, sigmas):
                    img_filtered = filters.gaussian(img, sigma=sigma, truncate=kernel_size, channel_axis=-1)
                    img_filtered = (img_filtered * 255).astype(np.uint8)
                    save_path = os.path.join(gaussian_dir, f'kernel_{kernel_size}_sigma_{sigma}', subdir, filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    io.imsave(save_path, img_filtered)

def prepare_submission(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)

class CustomDataset(Dataset):
    def __init__(self, gaussian_dir, target_dir, transform=None, train = True):
        self.gaussian_dir = gaussian_dir
        self.target_dir = target_dir
        self.transform = transform
        gaussian_subdir_path_1 = os.path.join(gaussian_dir, f'kernel_3_sigma_0.3')
        gaussian_subdir_path_2 = os.path.join(gaussian_dir, f'kernel_7_sigma_1')
        gaussian_subdir_path_3 = os.path.join(gaussian_dir, f'kernel_11_sigma_1.6')
        self.image_paths = []
        self.target_paths = []
        filenames = [i for i in range(100)]
        for i in range(len(filenames)):
            if i<10:
                filenames[i] = "0000000"+str(filenames[i])+".png"
            else:
                filenames[i] = "000000"+str(filenames[i])+".png"

        # Iterate over all subfolders (0 to 239)
        if train:
            img_nums = range(192)
        else:
            img_nums = range(192, 240)
        for subdir in img_nums:
            subdir = str(subdir)
            if len(subdir) == 2:
                subdir = "0" + subdir
            elif len(subdir) == 1:
                subdir = "00" + subdir
            target_subdir_path = os.path.join(target_dir, subdir)

            # Ensure subdirectories exist
            if os.path.isdir(gaussian_subdir_path_1) and os.path.isdir(gaussian_subdir_path_2) and os.path.isdir(gaussian_subdir_path_3) and os.path.isdir(target_subdir_path):
                # Iterate over all image files (00000000.png to 00000099.png)
                for filename in sorted(filenames):
                    img_id = os.path.splitext(filename)[0]
                    img_paths = [
                        os.path.join(gaussian_subdir_path_1, subdir, f'{img_id}.png'),
                        os.path.join(gaussian_subdir_path_2, subdir, f'{img_id}.png'),
                        os.path.join(gaussian_subdir_path_3, subdir, f'{img_id}.png')
                    ]
                    target_path = os.path.join(target_subdir_path, filename)

                    # Ensure all three blurred images and the target image exist
                    if all([os.path.exists(path) for path in img_paths]) and os.path.exists(target_path):
                        self.image_paths.extend(img_paths)
                        self.target_paths.append(target_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_paths = self.image_paths[idx]
        target_path = self.target_paths[idx//3]

        blurred_image = Image.open(img_paths)

        target_image = Image.open(target_path)

        if self.transform:
            blurred_image = self.transform(blurred_image)
            target_image = self.transform(target_image)

        return blurred_image, target_image


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    flag = False
    last_checkpoint_name = None
    best_test_loss = float('inf')
    epoch_test_losses = []
    epoch_train_losses = []

    for epoch in range(num_epochs):
        checkpoint_name = f"./checkpoints/epoch{epoch+1}.pth"
        if os.path.exists(checkpoint_name):
            last_checkpoint_name = checkpoint_name
        else:
            if flag == False:
                if last_checkpoint_name != None:
                    model.load_state_dict(torch.load(last_checkpoint_name))
                flag = True
            running_loss = 0.0
            total_epoch_loss = 0.0
            for i, (blurred_images, target_images) in enumerate(train_loader):
                blurred_images = blurred_images.to(device)
                target_images = target_images.to(device)

                optimizer.zero_grad()

                outputs = model(blurred_images)

                loss = criterion(outputs, target_images)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i+1)%10==0:
                    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                    total_epoch_loss += running_loss
                    running_loss = 0.0
            prepare_submission(model, checkpoint_name)
            test_loss = get_test_loss(model, test_loader, criterion, device, epoch)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Test Loss : {test_loss}')
            epoch_test_losses.append(test_loss)
            epoch_train_losses.append(total_epoch_loss)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                prepare_submission(model, "./checkpoints/best_model.pth")
    plt.plot(range(1, num_epochs+1), epoch_train_losses, label='Train loss')
    plt.plot(range(1, num_epochs+1), epoch_test_losses, label='Test loss')
    plt.savefig('loss.png')
    plt.close("all")

def get_test_loss(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for blurred_images, target_images in test_loader:
            blurred_images = blurred_images.to(device)
            target_images = target_images.to(device)
            outputs = model(blurred_images)
            loss = criterion(outputs, target_images)
            test_loss += loss.item()
        return test_loss / len(test_loader)


def main(sharp_images_dir, downscaled_dir, gaussian_dir, checkpoint_dir, device, num_epochs=10, batch_size=16, lr=0.001):
    # Paths to data directories
    rval = os.system(f'ls {sharp_images_dir} > /dev/null 2> /dev/null')
    if rval != 0:
        os.system('unzip train_sharp.zip > /dev/null 2> /dev/null')

    os.system(f'mkdir -p {downscaled_dir}')
    os.system(f'mkdir -p {gaussian_dir}')
    os.system(f'mkdir -p {checkpoint_dir}')

    # Step 2
    downscale_images(sharp_images_dir, downscaled_dir)
    logging.info("Images Downscaled")

    # Step 3
    apply_gaussian_filters(downscaled_dir, gaussian_dir)
    logging.info("Gaussian Filters Applied")

    # Define transformation
    transform = ToTensor()

    # Create custom dataset
    train_dataset = CustomDataset(gaussian_dir, downscaled_dir, transform, train = True)
    test_dataset = CustomDataset(gaussian_dir, downscaled_dir, transform, train = False)
    logging.info("Datasets Created")

    # Create data loaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    logging.info("Data Loaders Created")

    # Define model, loss function, optimizer
    model = UNet(3, 3)
    logging.info("Model Created")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set GPU or CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train the model
    logging.info("Starting Training")
    train_model(model, train_loader, test_loader, criterion, optimizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--sharp_images_dir', type=str, default='./train/train_sharp', help='Path to sharp images directory')
    parser.add_argument('--downscaled_dir', type=str, default='./downscaled_images', help='Path to store downscaled images directory')
    parser.add_argument('--gaussian_dir', type=str, default='./gaussian_images', help='Path to store gaussian images directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to store model checkpoints')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training (cpu or cuda:0)')
    args = parser.parse_args()
    logging.info("Starting process")
    main(sharp_images_dir=args.sharp_images_dir, downscaled_dir = args.downscaled_dir, gaussian_dir=args.gaussian_dir, checkpoint_dir=args.checkpoint_dir, device=args.device, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr)