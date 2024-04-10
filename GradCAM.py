from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import torch.nn as nn

import numpy as np
import csv
from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from metrics import BinaryClassificationMetrics
import pandas as pd
import torch
import torch.nn.functional as F

import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pdb
import wandb
import csv
import os
import shutil
import cv2
from torchvision.utils import save_image

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
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32])

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

    # PASCAL VOC Options
    # parser.add_argument("--year", type=str, default='2012',
                        # choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

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
    class LastConvLayerModel(nn.Module):
        def __init__(self):
            super(LastConvLayerModel, self).__init__()
            self.layers = list(list(model.children())[0].children())
            self.conv_layer_output = None
            # print("layers", self.layers)
            # print("layers", self.layers[-1])
        def forward(self, images):
            x = self.layers[0](images)
            for i, layer in enumerate(self.layers[1:-1]):
                x = layer(x)
            # print(x['low_level'].shape, x['out'].shape)
            llf = self.layers[-1].project(x['low_level'])  # Apply the project layer of DeepLabV3Head
            out = self.layers[-1].aspp(x['out']) 
            out = F.interpolate(out, size=llf.shape[2:], mode='bilinear', align_corners=False)    
            cf = torch.cat([llf,out], dim=1)
            x = F.relu(self.layers[-1].classifier.conv1(cf))
            x = F.relu(self.layers[-1].classifier.conv2(x))
            self.conv_layer_output = x
            x = self.layers[-1].classifier.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.layers[-1].classifier.fc1(x))
            x = F.relu(self.layers[-1].classifier.fc2(x))
            # print(x.shape)
            x = self.layers[-1].classifier.fc3(x)
            # print(self.conv_layer_output.shape)
            return x
    conv_model = LastConvLayerModel()
    conv_model = conv_model.to(device)
    binary_metrics.reset()
    ret_samples = []
    binary_criterion = nn.CrossEntropyLoss()
    TP = 0  # True Positive count
    TN = 0  # True Negative count
    FP = 0  # False Positive count
    FN = 0  # False Negative count
    
    csv_file_path = "validation15k.csv"
    fp_folder_path = "val_fp"
    fn_folder_path = "val_fn"
    tn_folder_path = "val_tn"
    tp_folder_path = "val_tp"
    os.makedirs(fp_folder_path, exist_ok=True)
    os.makedirs(fn_folder_path, exist_ok=True)
    os.makedirs(tn_folder_path, exist_ok=True)
    os.makedirs(tp_folder_path, exist_ok=True)
    with open(csv_file_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image Name", "Predicted Label", "Target Label"])
        # print(len(loader.dataset.images))
        for i, (images, labels, img_name) in tqdm(enumerate(loader)):
            # print("images", len(images))
            images = images.to(device)
            pred = conv_model(images)
            # print(pred.shape, labels.shape)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            loss = binary_criterion(pred, labels.view(1))
            targets = labels.cpu().numpy()
            binary_preds =pred.argmax(dim=-1).item()
            binary_metrics.update(targets, binary_preds)
            from torch import autograd
            grads = autograd.grad(pred[:, pred.argmax().item()], conv_model.conv_layer_output)
            # print("grads[0] shape", grads[0].shape)
            pooled_grads = grads[0].mean((0,2,3))
            # print("pooled_grads",pooled_grads.shape)
            conv_output = conv_model.conv_layer_output.squeeze()
            conv_output = F.relu(conv_output)
            # print("conv output", conv_output.shape)
            for i in range(len(pooled_grads)):
                conv_output[i,:,:] *= pooled_grads[i]
            heatmap = conv_output.mean(dim=0).squeeze()
            heatmap = heatmap / torch.max(heatmap)
            heatmap_np = heatmap.detach().cpu().numpy()
            heatmap_np = cv2.resize(heatmap_np, (images.shape[3], images.shape[2]))
            # print(heatmap_np.shape)
            heatmap_np = np.uint8(255 * heatmap_np)
            heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            image_np = images.permute(0, 2, 3, 1).cpu().numpy()[0]
            # print(image_np.shape)
            image_np = np.uint8(255 * image_np)
            # print(image_np.shape)
            overlay = cv2.addWeighted(heatmap_colored, 0.3, image_np, 0.7, 0)
            overlay_pil = Image.fromarray(overlay)
            overlay_image_name = os.path.splitext(os.path.basename(img_name[0]))[0] + '_overlay.jpg'
            img = os.path.splitext(os.path.basename(img_name[0]))[0]
            csv_writer.writerow([img, binary_preds, targets[0][0]])
            if binary_preds == targets[0][0] and targets[0][0]==1:
                shutil.copy(img_name[0], os.path.join(tp_folder_path, f"{img}.jpg"))
                overlay_pil.save(os.path.join(tp_folder_path, overlay_image_name))
            if binary_preds == targets[0][0] and targets[0][0]==0:
                shutil.copy(img_name[0], os.path.join(tn_folder_path, f"{img}.jpg"))
                overlay_pil.save(os.path.join(tn_folder_path, overlay_image_name))
            elif binary_preds == 1 and targets[0][0] == 0:
                shutil.copy(img_name[0], os.path.join(fp_folder_path, f"{img}.jpg"))
                overlay_pil.save(os.path.join(fp_folder_path, overlay_image_name))
            else:
                shutil.copy(img_name[0], os.path.join(fn_folder_path, f"{img}.jpg"))
                overlay_pil.save(os.path.join(fn_folder_path, overlay_image_name))

        metrics = binary_metrics.get_results()
        # Log individual metrics
        wandb.log({"Accuracy": metrics["Accuracy"]})
        wandb.log({"Precision": metrics["Precision"]})
        wandb.log({"Recall": metrics["Recall"]})
        wandb.log({"F1 Score": metrics["F1 Score"]})
        wandb.log({"Validation loss": loss})
        wandb.log({"True Positive": metrics["TP"]})
        wandb.log({"True Negative": metrics["TN"]})
        wandb.log({"False Positive":metrics["FP"]})
        wandb.log({"False Negative": metrics["FN"]})

    score = binary_metrics.get_results()
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 2
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 2 #Binary classification


    # Initialize wandb
    wandb.init(project="deeplabv3plus_mord_bin_clf", name="emarg_clf_ckpt-14-imageblend+cutmix", config=vars(opts))


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

    # # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    # # print(model)
    checkpoint = torch.load('/home/shubhamp/Downloads/Segmentation_models/DeepLabV3Plus_Emarg15k/checkpoints_dlv3+Seg15k_6classes/best_deeplabv3plus_resnet101_cityscapes_os8.pth')['model_state']   
    backbone_model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    backbone_keys = ['backbone.' + k for k in backbone_model.state_dict().keys()]
    checkpoint_filtered = {k: v for k, v in checkpoint.items() if any(k.startswith(prefix) for prefix in backbone_keys)}
    backbone_model.load_state_dict(checkpoint_filtered, strict=False)
    # Transfer backbone weights to the classifier
    model.backbone.load_state_dict(backbone_model.backbone.state_dict())
    
    # Freeze Backbbone
    # model.backbone.eval()
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    # Print information about the frozen backbone
    # print("Backbone Frozen: ", all(not param.requires_grad for param in model.backbone.parameters()))

    # Set up binary classification metrics
    binary_metrics = BinaryClassificationMetrics()

    # Set up optimizer

    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # optimizer = torch.optim.SGD(params=model.classifier.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    # if opts.loss_type == 'focal_loss':
    #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    # elif opts.loss_type == 'cross_entropy':
    #     criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

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
        # print(checkpoint["model_state"])

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
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
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
    binary_criterion = nn.CrossEntropyLoss()
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, img_name) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)

            # print("Image size:", images.shape)
            # print("Label size:", labels.shape)

            optimizer.zero_grad()
            outputs = model(images)
            # print("shape:",outputs.shape)
            # print(torch.argmax(outputs, dim=1).unsqueeze(1).shape)
            # print(labels, torch.argmax(outputs, dim=1).unsqueeze(1))
            # loss = binary_criterion(torch.argmax(outputs, dim=1).unsqueeze(1).float(), labels)  # binary loss
            
            loss= binary_criterion(outputs, labels.squeeze())
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
