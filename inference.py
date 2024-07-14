import torch
from torchvision.transforms import ToTensor, ToPILImage
import os
from PIL import Image
from unet_model import UNet
import argparse


def evaluate(saved_model_path = "./checkpoints/best_model.pth", blur_image_dir = "./mp2_test/custom_test/blur", sharp_image_dir = "./mp2_test/custom_test/sharp", test_script_path = "./mp2_test/eval.py", device = torch.device("cpu")):
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(saved_model_path))
    model.cuda(device)
    for input_file_name in os.listdir(blur_image_dir):
        if os.path.isfile(os.path.join(blur_image_dir, input_file_name)) and (input_file_name.endswith('.png') or input_file_name.endswith('.jpg')):
            # Load the image
            input_image = Image.open(os.path.join(blur_image_dir, input_file_name))

            # Preprocess the image
            transform = ToTensor()
            input_tensor = transform(input_image.resize((448, 256))).unsqueeze(0)
            input_tensor = input_tensor.to(device)

            # Deblur the image using the model
            with torch.no_grad():
                model.eval()
                output_tensor = model(input_tensor)

            # Convert the output tensor to an image
            output_image = ToPILImage()(output_tensor.squeeze(0))

            # Save the deblurred image
            output_image.save(os.path.join(sharp_image_dir, input_file_name))
    # Run eval.py to calculate PSNR
    os.system("python3 " + test_script_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model', type=str, default="./checkpoints/best_model.pth", help='Path to saved model')
    parser.add_argument('--blur_dir', type=str, default="./mp2_test/custom_test/blur", help='Directory containing blur images')
    parser.add_argument('--sharp_dir', type=str, default="./mp2_test/custom_test/sharp", help='Directory to save sharp images')
    parser.add_argument('--test_script', type=str, default="./mp2_test/eval.py", help='Path to test script')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluate(saved_model_path=args.model, blur_image_dir=args.blur_dir, sharp_image_dir=args.sharp_dir, device=device)
