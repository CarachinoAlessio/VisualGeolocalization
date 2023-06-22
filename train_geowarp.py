"""
Contains training code for GeoWarp and it is divided as following:
- modules import 
- create geowarp model: FeatureExtractor (based on CNN) + HomographyRegression
- create training dataset (small is SF-XS)
- use optimizer and learning rate
- start training for 3 epochs. During each epoch, for each example batch, model does a forward propagation and calculate the loss to use the backward
propagation and update model weights. then the accuracy of the batch is calculated
- at the end of the training phase, the model is saved 
"""
#imports
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from torchvision.transforms.functional import hflip
import os
import test
import util
import parser
import commons
import cosface_loss
import arcface_loss
import sphereface_loss 
import augmentations
from model import network

#import dataset_qp  # Used for weakly supervised losses, it yields query-positive pairs
import dataset_warp  # Used to train the warping regressiong module in a self-supervised fashion
#import dataset_geoloc  # Used for testing

### FROM TRAIN.PY ### 
#needed
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


def to_cuda(list_):
    """Move to cuda all items of the list."""
    return [item.cuda() for item in list_]

def compute_loss(loss, weight):
    """Compute loss and gradients separately for each loss, and free the
    computational graph to reduce memory consumption.
    """
    loss *= weight
    loss.backward()
    return loss.item()

# SAME AS TRAIN.PY
torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

# MODEL
'''
Since we are using Geo Warp model has to be different
'''
# train.py of geo_warp
############### MODEL ###############
features_extractor = network.FeatureExtractor(args.backbone, args.fc_output_dim)
global_features_dim = commons.get_output_dim(features_extractor, "gem")

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

# now there should be the resume_model but of course it is different
if args.resume_model is not None: 
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    features_extractor.load_state_dict(model_state_dict)
    del model_state_dict
else:
    logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                        "Feature Extractor is not initialized!")

homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1)
model = network.GeoWarp(features_extractor, homography_regression).cuda().eval()
model = torch.nn.DataParallel(model)

############### DATASETS & DATALOADERS ###############
#No optimizer atm
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)] #same as cos/arc/sphere face
#ss_dataset = dataset_warp.HomographyDataset(root_path=f"{args.datasets_folder}/{args.dataset_name}/images/train", k=args.k) -> original
ss_dataset = [dataset_warp.HomographyDataset(args, args.train_set_folder, M=args.M, N=args.N, current_group=n, min_images_per_class=args.min_images_per_class, k=args.k) for n in range(args.groups_num)] # k = parameter k, defining the difficulty of ss training data, default = 0.6    

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold,args=args)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold,args=args)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

## OPTIMIZER AND LOSS
criterion = torch.nn.CrossEntropyLoss() # same as cosplace
mse = torch.nn.MSELoss() #used in the original repo
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.optim == "adam":
    optim = torch.optim.Adam(homography_regression.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optim = torch.optim.SGD(homography_regression.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#from train.py
logging.info(f"Using {args.loss_function} loss function.")
if args.loss_function == "cosface":
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups] 
elif args.loss_function == "sphereface": 
        classifiers = [sphereface_loss.SphereFaceLoss(args.fc_output_dim, len(group)) for group in groups]
elif args.loss_function == "arcface":
        classifiers = [arcface_loss.ArcFaceLoss(args.fc_output_dim, len(group)) for group in groups]
else:
        logging.info(f"OUCH! Please provide the loss function with --loss_function [cosface - sphereface - arcface]")
        logging.info(f"Setting cosface...")
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups] 

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

#### Resume
if args.resume_train: #starting from checkpoint
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else: #starting from zero
    best_val_recall1 = start_epoch_num = 0

if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

# START TRAINING
for epoch_num in range(start_epoch_num, args.epochs_num):
#### Train
    epoch_start_time = datetime.now() #rn date
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    # batch size = 32,same as cosplace to iterate in the dataset
    dataloader_iterator = iter(dataloader)
    # now i use the new one
    # same things but for the two different datasets
    ss_dataloader = commons.InfiniteDataLoader(ss_dataset[current_group_num], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=True)
    ss_data_iter = iter(ss_dataloader) 

    model=model.train() # training mode
    #epoch_losses = np.zeros((0, 3), dtype=np.float32)
    #epoch_losses = np.zeros((0, 1), dtype=np.float32)
    epoch_losses = np.zeros((0, 2), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _ = next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        #warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = to_cuda(next(ss_data_iter))
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = next(ss_data_iter)   # dal warping dataset prende le due immagini warped e i due punti delle intersezioni
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = warped_img_1.to(args.device), warped_img_2.to(args.device), warped_intersection_points_1.to(args.device), warped_intersection_points_2.to(args.device)  # warping dataset

        with torch.no_grad():
            similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])
        
        optim.zero_grad()
        model_optimizer.zero_grad()                                        # setta il gradiente a zero per evitare double counting (passaggio classico dopo ogni iterazione)
        classifiers_optimizers[current_group_num].zero_grad() 

        if not args.use_amp16:
            descriptors = model("features_extractor", [images, "global"])
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            loss.backward()
            loss=loss.item()
            del output, images
        # ss_loss
            if args.ss_w != 0:
                pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
                pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
                ss_loss = (mse(pred_warped_intersection_points_1[:, :4], warped_intersection_points_1) +
                            mse(pred_warped_intersection_points_1[:, 4:], warped_intersection_points_2) +
                            mse(pred_warped_intersection_points_2[:, :4], warped_intersection_points_2) +
                            mse(pred_warped_intersection_points_2[:, 4:], warped_intersection_points_1))
                #ss_loss = compute_loss(ss_loss, args.ss_w)
                ss_loss.backward()
                ss_loss=ss_loss.item()
                del pred_warped_intersection_points_1, pred_warped_intersection_points_2
            else:
                ss_loss = 0
            epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]]))) # both losses
            del loss, ss_loss 
            #step update
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
            optim.step()
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model("features_extractor", [images, "global"])                
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()

            #remember to concatenate both losses
            epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]]))) # both losses
            del loss, ss_loss
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    
    ### Evaluation (still in the for - delete the ())
    recalls, recalls_str, _ = test.use_geowarp(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, output_folder)
    #### end of evaluation
    ### end of training. 

# TEST
logging.info(f"The training is over in {str(datetime.now() - start_time)[:-7]}, now it's test time")
#### Test best model on test set v1
best_model_state_dict = torch.load(f"{output_folder}/best_model.pth") #output_folder model
model.load_state_dict(best_model_state_dict) #try best model on queries_v1

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str, predictions = test.use_geowarp(args, test_ds, model)
logging.info(f"{test_ds}: {recalls_str}")

_, reranked_test_recalls_pretty_str = test.use_rerank(model, predictions, test_ds, num_reranked_predictions = args.num_reranked_preds) # num_reranked_predictions=5 by def
logging.info(f"test with no warping: {recalls_str}")
logging.info(f"test after warping - {reranked_test_recalls_pretty_str}")
logging.info("Experiment finished (without any errors)")
