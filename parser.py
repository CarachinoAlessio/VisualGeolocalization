import os
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Weight Decay
    parser.add_argument("--wd", type=float, default=None, help="Define the weight decay of the optimizer")
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    # Backbone
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152", "convnext_tiny",
                                 "efficientnet_v2_s"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    # Training parameters
    parser.add_argument("--loss_function", type=str, help="choose the loss function [cosface, sphereface, arcface]",
                        default="cosface"),
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=50, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default=None,
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    # GeoWarp parameters
    parser.add_argument("--k", type=int, default=0.6,
                        help="parameter k, defining the difficulty of ss training data")
    parser.add_argument("--ss_w", type=float, default=1,
                        help="weight of self-supervised loss")
    parser.add_argument("--consistency_w", type=float, default=0.1,
                        help="weight of consistency loss")
    parser.add_argument("--features_wise_w", type=float, default=10,
                        help="weight of features-wise loss")
    parser.add_argument("--qp_threshold", type=float, default=1.2,
                        help="Threshold distance (in features space) for query-positive pairs")
    parser.add_argument("--num_reranked_preds", type=int, default=5,
                        help="number of predictions to re-rank at test time")
    parser.add_argument("--kernel_sizes", nargs='+', default=[7, 5, 5, 5, 5, 5],
                        help="size of kernels in conv layers of Homography Regression")
    parser.add_argument("--channels", nargs='+', default=[225, 128, 128, 64, 64, 64, 64],
                        help="num channels in conv layers of Homography Regression")
    # Multi scale parameters
    parser.add_argument("--multi_scale", action='store_true', help="Use multi scale")
    parser.add_argument("--select_resolutions", type=float, default=[0.526, 0.588, 1, 1.7, 1.9], nargs="+",
                        help="Usage: --select_resolution 1 2 4 6")
    parser.add_argument("--multi_scale_method", type=str, default="avg", choices=["avg", "sum", "max", "min"],
                        help="Usage:--multi_scale_method=avg")
    # Domain adaptation parameters & Data augmentation
    parser.add_argument("--grl_param", default=None, type=float,
                        help="Use Gradient Reversal Layer (GRL) initialized with the specified param")
    parser.add_argument("--night_test", type=bool, default=False, help="To be enabled when domain is tokyo night")
    parser.add_argument("--night_brightness", type=float, default=0.1,
                        help="Brightness of augmented train data when testing on night domain")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Directory of source dataset, for example: '/content/small/train/'")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="Directory of target dataset, for example: '/content/night_target/'")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        help="This includes pre/post-processing methods and prediction refinement")
    # Optimizer
    parser.add_argument("--optim", type=str, default="adam", help="Adam is the default choice", choices=["adam", "sgd"])
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")

    parser.add_argument("--grl_model_path")
    parser.add_argument("--geowarp_model_path")
    parser.add_argument("--ensemble_merge_preds", type=bool, default=False, help="Specify which tecnique has to be "
                                                                                 "used in case of ensembler with GRL "
                                                                                 "and GeoWarp."
                                                                                 " NIF is selected by default.")
    args = parser.parse_args()

    if args.dataset_folder is None:
        try:
            args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
        except KeyError:
            raise Exception("You should set parameter --dataset_folder or export " +
                            "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                            "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")

    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")

    if is_training:
        args.train_set_folder = os.path.join(args.dataset_folder, "train")
        if not os.path.exists(args.train_set_folder):
            raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")

        args.val_set_folder = os.path.join(args.dataset_folder, "val")
        if not os.path.exists(args.val_set_folder):
            raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")

    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    if not os.path.exists(args.test_set_folder):
        raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")

    return args