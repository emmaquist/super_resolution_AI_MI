import argparse
import os.path
import numpy as np
import torch.cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset.data_loader import Dataloader
from dataset.utils import split_data
from torchmetrics import StructuralSimilarityIndexMeasure, MeanSquaredError

import torch.backends.cudnn
from discrete_vae import DiscreteVAE
import torch.nn as nn
import torch.nn.functional as F

def create_data_sets(
    source_directory, target_directory
) :
    """
    Creates the two separate datasets if necessary
    :param source_directory: Path to source images
    :param target_directory: Path to target images
    :return: Dataloader objects
    """
    if "/test/" in source_directory:
        test_set = Dataloader(source_directory, target_directory)
        return test_set, []

    train_source_dir, val_source_dir = split_data(source_directory)
    train_target_dir, val_target_dir = split_data(target_directory)
    train_set = Dataloader(train_source_dir, train_target_dir)
    val_set = Dataloader(val_source_dir, val_target_dir)

    return train_set, val_set


def create_torch_data_loaders(
    train_set, val_set, batch_size = 16, workers = 8,
):
    """
    Creates torch DataLoader classes using the separate path folder for the images
    :param train_set: Path to training set
    :param val_set: Path to validation set
    :return: Two DataLoader objects
    """
    train_torch_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, num_workers=workers, pin_memory=True)
    val_torch_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=workers, pin_memory=True)
    return train_torch_loader, val_torch_loader


def train(args):
    torch.manual_seed(args.seed)

    # When running on cpu
    if args.device == "cpu":
        args.device = torch.device("cpu")

    # Spare some memory, since we are using conv layers
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    print(torch.device)
    # Init model
    model = DiscreteVAE()
    model = model.to(args.device)

    # Loss and metrics
    loss_fn_mse = torch.nn.MSELoss(reduction="mean")
    mse = MeanSquaredError().to(args.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=255).to(args.device)
    best_ssim = 0

    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # DataLoaders
    train_dataset, val_dataset = create_data_sets(
        source_directory=args.source_directory, target_directory=args.target_directory
    )
    train_loader, val_loader = create_torch_data_loaders(
        train_set=train_dataset, val_set=val_dataset, batch_size=args.batch_size, workers=args.workers,
    )
    test_dataset, val_test = create_data_sets(source_directory=args.source_directory_test, target_directory=args.target_directory_test)
    test_loader, _ = create_torch_data_loaders(test_dataset, val_test, batch_size=args.batch_size, workers=args.workers)
    if not args.only_test:
        for epoch in tqdm(range(args.num_epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            model.train()
            for i, (images, labels) in enumerate(tqdm(train_loader, desc="Epoch {}".format(epoch))):

                # Slice 3D to 2D
                for s in range(images.shape[1] - 1):
                    img_1 = images[:, s, :, :].to(args.device)
                    img_2 = images[:, s + 1, :, :].to(args.device)
                    label = labels[:, s * 2 + 1, :, :].to(args.device)
                    img_1 = img_1[None, :, :, :]
                    img_2 = img_2[None, :, :, :]


                    optimizer.zero_grad()

                    kl_loss, output = model(img_1, img_2)

                    loss = loss_fn_mse(output[0][0], label[0])
                    running_loss += loss.detach().item()

                    # backpropagate the loss
                    loss.backward()
                    # adjust parameters based on the calculated gradients
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    optimizer.step()
            scheduler.step(epoch)

            # Calculate validation metrics
            with torch.no_grad():
                model.eval()
                val_ssim = []
                val_mse = []
                for j, (images_val, labels_val) in enumerate(val_loader):
                    for s in range(images_val.shape[1] - 1):

                        img_1_val = images_val[:, s, :, :].to(args.device)
                        img_2_val = images_val[:, s + 1, :, :].to(args.device)

                        label_val = labels_val[:, s * 2 + 1, :, :].to(args.device)

                        img_1_val = img_1_val[None, :, :, :]
                        img_2_val = img_2_val[None, :, :, :]
                        _, output_val = model.forward(img_1_val, img_2_val)
                        if s == 0:
                            outputs_val = output_val.reshape(1, 256, 256)
                            inbetween_labels_val = label_val.reshape(1, 256, 256)
                        else:
                            outputs_val = torch.cat((outputs_val, output_val.reshape(1, 256, 256)), 0)
                            inbetween_labels_val = torch.cat((inbetween_labels_val, label_val.reshape(1, 256, 256)), 0)


                    outputs_val = outputs_val.reshape(1, 95, 256, 256).detach().cpu()
                    inbetween_labels_val = inbetween_labels_val.reshape(1, 95, 256, 256).detach().cpu()
                    val_mse.append(mse(outputs_val, inbetween_labels_val).cpu().item())
                    val_ssim.append(ssim(outputs_val, inbetween_labels_val).cpu().item())

                val_ssim = torch.FloatTensor(val_ssim)
                val_mse = torch.FloatTensor(val_mse)
                if torch.mean(val_ssim) > best_ssim:
                    print("model_saved")
                    best_ssim = torch.mean(val_ssim)
                    torch.save(model.state_dict(), "best_model.pth")

            if args.log:
                writer.add_scalar("Loss training (epoch)", running_loss / (i + 1), epoch)
                writer.add_image("input image", images, epoch, dataformats="NCHW")
                writer.add_image("reconstruction slice", np.array(output), epoch, dataformats="NCHW")
                writer.add_image("expected output", labels, epoch, dataformats="NCHW")
            else:
                plt.imsave("results/output_{}_80.png".format(epoch), outputs_val[0][40].detach().cpu().numpy(),
                           cmap='gray', vmin=0, vmax=255)
                plt.imsave("results/output_{}_81.png".format(epoch), outputs_val[0][41].detach().cpu().numpy(),
                           cmap='gray', vmin=0, vmax=255)
                plt.imsave("results/expected_{}_80.png".format(epoch), inbetween_labels_val[0][40].detach().cpu().numpy(), cmap='gray',
                           vmin=0, vmax=255)
                plt.imsave("results/expected_{}_81.png".format(epoch), inbetween_labels_val[0][41].detach().cpu().numpy(), cmap='gray',
                           vmin=0, vmax=255)
                with open("total_result_map/results/log/loss.txt", 'a') as outfile:
                    try:
                        outfile.write("Epoch {} loss: {}\n".format(epoch, running_loss / i))
                    except ZeroDivisionError:
                        outfile.write("Epoch {} loss: {}\n".format(epoch, running_loss / (i + 1)))
                    outfile.write("Epoch {} MSE: {}\n".format(epoch, torch.mean(val_mse)))
                    outfile.write("Epoch {} SSIM: {}\n".format(epoch, torch.mean(val_ssim)))
                    print("Loss: ", running_loss / (i + 1))
                    print("MSE: ", torch.mean(val_mse))
                    print("SSIM: ", torch.mean(val_ssim))
                    outfile.write("\n")

    with torch.no_grad():
        model = DiscreteVAE().to(args.device)
        model.load_state_dict(torch.load("best_model_51.pth"))
        test_ssim = []
        test_mse = []
        model.eval()

        for j, (images_test, labels_test) in enumerate(test_loader):
            for s in range(images_test.shape[1] - 1):
                img_1_test = images_test[:, s, :, :].to(args.device)
                img_2_test = images_test[:, s + 1, :, :].to(args.device)
                label_test = labels_test[:, s * 2 + 1, :, :].to(args.device)

                img_1_test = img_1_test[None, :, :, :]
                img_2_test = img_2_test[None, :, :, :]
                _, output_test = model.forward(img_1_test, img_2_test)
                if s == 0:
                    outputs_test = output_test.reshape(1, 256, 256)
                    img_1_test = img_1_test.reshape(1, 256, 256)
                    inbetween_labels_test = label_test.reshape(1, 256, 256)
                    all_outputs_test = torch.cat((img_1_test, outputs_test), 0)
                    all_labels_test = torch.cat((img_1_test, inbetween_labels_test), 0)
                else:

                    outputs_test = torch.cat((outputs_test, output_test.reshape(1, 256, 256)), 0)
                    inbetween_labels_test = torch.cat((inbetween_labels_test, label_test.reshape(1, 256, 256)), 0)
                    all_outputs_test = torch.cat(
                        (all_outputs_test, img_1_test.reshape(1, 256, 256), output_test.reshape(1, 256, 256)), 0)
                    all_labels_test = torch.cat(
                        (all_labels_test, img_1_test.reshape(1, 256, 256), label_test.reshape(1, 256, 256)), 0)

            outputs_test = outputs_test.reshape(1, 95, 256, 256).detach().cpu()
            inbetween_labels_test = inbetween_labels_test.reshape(1, 95, 256, 256).detach().cpu()
            all_outputs_test = all_outputs_test.reshape(1, 190, 256, 256).detach().cpu()
            all_labels_test = all_labels_test.reshape(1, 190, 256, 256).detach().cpu()
            test_mse.append(mse(outputs_test, inbetween_labels_test).cpu().item())
            test_ssim.append(ssim(outputs_test, inbetween_labels_test).cpu().item())
    if args.log:
        writer.add_scalar("Loss training (epoch)", running_loss / (i + 1), epoch)
        writer.add_image("input image", images, epoch, dataformats="NCHW")
        writer.add_image("reconstruction slice", np.array(output), epoch, dataformats="NCHW")
        writer.add_image("expected output", labels, epoch, dataformats="NCHW")
    else:
        plt.imsave("results/test_output_128.png", outputs_test[0][:, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_output_129.png", outputs_test[0][:, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_output_128_.png", outputs_test[0][:, :, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_output_129_.png", outputs_test[0][:, :, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_output_80.png", outputs_test[0][40].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_output_81.png", outputs_test[0][41].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)

        plt.imsave("results/test_expected_128.png", inbetween_labels_test[0][:, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_expected_129.png", inbetween_labels_test[0][:, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_expected_128_.png", inbetween_labels_test[0][:, :, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_expected_129_.png", inbetween_labels_test[0][:, :, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_expected_40.png", inbetween_labels_test[0][40].detach().cpu().numpy(), cmap='gray',
                   vmin=0, vmax=255)
        plt.imsave("results/test_expected_41.png", inbetween_labels_test[0][41].detach().cpu().numpy(), cmap='gray',
                   vmin=0, vmax=255)

        plt.imsave("results/test_all_output_128.png", all_outputs_test[0][:, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_output_129.png", all_outputs_test[0][:, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_output_128_.png", all_outputs_test[0][:, :, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_output_129_.png", all_outputs_test[0][:, :, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_output_80.png", all_outputs_test[0][80].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_output_81.png", all_outputs_test[0][81].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)

        plt.imsave("results/test_all_expected_128.png", all_labels_test[0][:, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_expected_129.png", all_labels_test[0][:, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_expected_128_.png", all_labels_test[0][:, :, 128].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_expected_129_.png", all_labels_test[0][:, :, 129].detach().cpu().numpy(),
                   cmap='gray', vmin=0, vmax=255)
        plt.imsave("results/test_all_expected_80.png", all_labels_test[0][80].detach().cpu().numpy(), cmap='gray',
                   vmin=0, vmax=255)
        plt.imsave("results/test_all_expected_81.png", all_labels_test[0][81].detach().cpu().numpy(), cmap='gray',
                   vmin=0, vmax=255)

        test_ssim = torch.FloatTensor(test_ssim)
        test_mse = torch.FloatTensor(test_mse)
        with open("total_result_map/results/log/loss.txt", 'a') as outfile:
            outfile.write("Test MSE: {}\n".format(torch.mean(test_mse)))
            outfile.write("Test SSIM: {}\n".format(torch.mean(test_ssim)))
            print("MSE: ", torch.mean(test_mse))
            print("SSIM: ", torch.mean(test_ssim))
            outfile.write("\n")

    return "DONE"


def main():
    if not os.path.exists("total_result_map/results"):
        os.makedirs("total_result_map/results")
    if not os.path.exists("total_result_map/results/log"):
        os.makedirs("total_result_map/results/log")
    parser = argparse.ArgumentParser(description="Arguments for training the VAE model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--only_test", type=bool, default=True)
    parser.add_argument("--device", type=str, default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    parser.add_argument(
        "-source_directory",
        type=str,
        default="dataset/data_train_plus_test_source_resolution/train/source_resolution",
    )
    parser.add_argument(
        "-target_directory",
        type=str,
        default="dataset/data_train_plus_test_source_resolution/train/target_resolution",
    )
    parser.add_argument(
        "-source_directory_test",
        type=str,
        default="dataset/data_train_plus_test_source_resolution/test/source_resolution",
    )
    parser.add_argument(
        "-target_directory_test",
        type=str,
        default="dataset/data_train_plus_test_source_resolution/test/target_resolution",
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    writer = SummaryWriter()
    main()