from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import BinaryClassificationMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pdb
import wandb


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='/home/shubhamp/Downloads/Clf_Emarg15k_val1500_Imageblend+cutmix',
                        help="path to Dataset")
    parser.add_argument("--active_list", type=str, default=None, help="path to Dataset")

    parser.add_argument("--save_path",type=str,default='CITY_768x768',help="name of folder to save checkpoint")

    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=200000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=3350) #model checkpoint save interval
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    # parser.add_argument("--loss_type", type=str, default='cross_entropy',
    #                     choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=100,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=3350,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            et.ExtResize(( 512,384 )),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(( 512,384 )),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root, split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root, split='val', transform=val_transform)
        
    return train_dst, val_dst

# UPDATED SAVE_VAL RESULTS
def validate(opts, model, loader, device, binary_metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""

    binary_metrics.reset()
    ret_samples = []
    binary_criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        # Open the file for writing
        with open("validation15k.txt", "w") as file:
            for i, (images, labels) in tqdm(enumerate(loader)):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)

                outputs = model(images)
                preds = torch.sigmoid(outputs)
                binary_preds = (preds > 0.5).float()  # Apply threshold for binary classification
                # print(binary_preds)
                binary_preds = binary_preds.cpu().numpy()
                targets = labels.cpu().numpy()
                loss = binary_criterion(outputs, labels)

                binary_metrics.update(targets, binary_preds)

                for j in range(len(images)):
                    img_path = loader.dataset.images[i * opts.val_batch_size + j]  # Corrected line

                    # Extract the actual image name from the path
                    img_name = os.path.splitext(os.path.basename(img_path))[0]

                    # Write the image name and predicted label to the file
                    file.write(f"Image Name: {img_name}, Predicted Label: {binary_preds[j][0]}\n")

                # Print the image name and predicted label
                # print(f"Image Name: {img_name}, Predicted Label: {preds[j][0]}")

            # Log metrics to WandB
            metrics = binary_metrics.get_results()
            
            # Log individual metrics
            wandb.log({"Accuracy": metrics["Accuracy"], "Step": i})
            wandb.log({"Precision": metrics["Precision"], "Step": i})
            wandb.log({"Recall": metrics["Recall"], "Step": i})
            wandb.log({"F1 Score": metrics["F1 Score"], "Step": i})
            wandb.log({"Validation loss": loss})

    score = binary_metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 2 #Binary classification


    # Initialize wandb
    wandb.init(project="classificationemarg15k_learnablebackbone_ImageBlend+CutMix2classes_Exp4", name="metric_plots", config=vars(opts))

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'cityscapes' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Define model with a suitable backbone
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)

    # Load pretrained backbone weights
    checkpoint = torch.load('/home/shubhamp/Downloads/Segmentation_models/DeepLabV3Plus_Emarg15k/checkpoints_Imageblend+CutMix_2classes/best_deeplabv3plus_resnet101_cityscapes_os8.pth')['model_state']
    backbone_model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    backbone_keys = ['backbone.' + k for k in backbone_model.state_dict().keys()]
    checkpoint_filtered = {k: v for k, v in checkpoint.items() if any(k.startswith(prefix) for prefix in backbone_keys)}
    backbone_model.load_state_dict(checkpoint_filtered, strict=False)

    # Transfer backbone weights to the classifier
    model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    # model.classifier.load_state_dict(backbone_model.classifier.state_dict())
    # print(model)

    
    # Freeze Backbbone
    # model.backbone.eval()
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    # # Print information about the frozen backbone
    # print("Backbone Frozen: ", all(not param.requires_grad for param in model.backbone.parameters()))

    # Set up binary classification metrics
    binary_metrics = BinaryClassificationMetrics()

    # Set up optimizer

    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # optimizer = torch.optim.SGD(params=model.classifier.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
 
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
    
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  #

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, binary_metrics=binary_metrics, ret_samples_ids=vis_sample_id)
        print(binary_metrics.to_str(val_score))
        return

    if not os.path.exists('checkpoints/'+opts.save_path):
        os.makedirs('checkpoints/'+opts.save_path)

    interval_loss = 0
    binary_criterion = nn.BCEWithLogitsLoss()
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # print("Image size:", images.shape)
            # print("Label size:", labels.shape)

            optimizer.zero_grad()
            outputs = model(images)
            # print("shape:",outputs.shape)

            loss = binary_criterion(outputs, labels)  # binary loss
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            # Add after the loss calculation
            wandb.log({"TrainingLoss": np_loss, "epochs": cur_epochs})

            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % opts.print_interval == 0:
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                  (cur_epochs, cur_itrs, opts.total_itrs, np_loss))

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/'+opts.save_path+'/latest_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, binary_metrics= binary_metrics,
                    ret_samples_ids=vis_sample_id)
                print(binary_metrics.to_str(val_score))
                print('Best score till now:', best_score)

                if val_score['Accuracy'] > best_score:  # save best model
                    best_score = val_score['Accuracy']
                    save_ckpt('checkpoints/'+opts.save_path+'/best_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))


                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Accuracy'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
        scheduler.step()

        if cur_itrs >= opts.total_itrs:
            return


if __name__ == '__main__':
    main()
