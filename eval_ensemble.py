import sys
import torch
import logging
import multiprocessing
from datetime import datetime

import test_ensemble
import parser
import commons
from model import network
from datasets.test_dataset import TestDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments(is_training=False)
start_time = datetime.now()
output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(output_folder, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

#### Model
model_grl = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
model_geowarp = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

features_extractor = network.FeatureExtractor(args.backbone, args.fc_output_dim) 
global_features_dim = commons.get_output_dim(features_extractor, "gem")    
homography_regression = network.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1) # inizializza il layer homography

model_geowarp = network.GeoWarp(features_extractor, homography_regression)
model_geowarp=torch.nn.DataParallel(model_geowarp)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.grl_model_path is not None:
    logging.info(f"Loading model from {args.grl_model_path}")
    model_state_dict = torch.load(args.grl_model_path)
    
    del model_state_dict["domain_discriminator.1.weight"]
    del model_state_dict["domain_discriminator.1.bias"]
    del model_state_dict["domain_discriminator.3.weight"]
    del model_state_dict["domain_discriminator.3.bias"]
    del model_state_dict["domain_discriminator.5.weight"]
    del model_state_dict["domain_discriminator.5.bias"]
    model_grl.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the grl model (--grl_model_path parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

if args.geowarp_model_path is not None:
    logging.info(f"Loading model from {args.geowarp_model_path}")
    model_state_dict = torch.load(args.geowarp_model_path)
    model_geowarp.load_state_dict(model_state_dict)
else:
    logging.info("WARNING: You didn't provide a path to resume the geowarp model (--geowarp_model_path parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

model_grl = model_grl.to(args.device)
model_geowarp = model_geowarp.to(args.device)

test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                      positive_dist_threshold=args.positive_dist_threshold, args=args)

recalls, recalls_str = test_ensemble.test(args, test_ds, model_grl, model_geowarp)
logging.info(f"{test_ds}: {recalls_str}")
