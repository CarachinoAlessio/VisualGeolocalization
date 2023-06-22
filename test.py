import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from dataset_warp import compute_warping

# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""

    model = model.eval()
    if args.multi_scale:
        # avg by default
        logging.info(f"Test with multi-scale, the multi-scale method is: {args.multi_scale_method}")
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            # descriptors = model(images.to(args.device))
            if args.multi_scale and args.multi_scale_method == 'avg':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists 	
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model(img.to(args.device))
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors = torch.mean(feature.type(torch.float32), dim=-1)
            elif args.multi_scale and args.multi_scale_method == 'sum':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists 	
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model(img.to(args.device))
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors = torch.sum(feature.type(torch.float32), dim=-1)
            elif args.multi_scale and args.multi_scale_method == 'max':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists 	
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model(img.to(args.device))
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors, max_index = torch.max(feature.type(torch.float32), dim=-1)
                del max_index
            elif args.multi_scale and args.multi_scale_method == 'min':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists 	
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model(img.to(args.device))
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors, min_index = torch.min(feature.type(torch.float32), dim=-1)
                del min_index
            else:
                descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str


# GeoWarp
def use_geowarp(args: Namespace, eval_ds: Dataset, model: torch.nn.Module):
    """Compute descriptors of the given dataset and compute the recalls."""

    model = model.eval()
    if args.multi_scale:
        # avg by default
        logging.info(f"Test with multi-scale, the multi-scale method is: {args.multi_scale_method}")
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")

        for images, indices in tqdm(database_dataloader, ncols=100):
            if args.multi_scale and args.multi_scale_method == 'avg':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model("features_extractor", [img, "global"])
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors = torch.mean(feature.type(torch.float32), dim=-1)
            elif args.multi_scale and args.multi_scale_method == 'sum':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model("features_extractor", [img, "global"])
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors = torch.sum(feature.type(torch.float32), dim=-1)
            elif args.multi_scale and args.multi_scale_method == 'max':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model("features_extractor", [img, "global"])
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors, max_index = torch.max(feature.type(torch.float32), dim=-1)
                del max_index
            elif args.multi_scale and args.multi_scale_method == 'min':
                H = args.resize[0]
                W = args.resize[1]
                HxW = args.resize
                original = images
                # create the resolution lists
                H_list = [int(H / i) for i in args.select_resolutions]
                W_list = [int(W / i) for i in args.select_resolutions]
                multi_scale = []
                for i, j in zip(H_list, W_list):
                    size = (i, j)  # size resolution of the resize
                    tra = torch.nn.Sequential(transforms.Resize(size))  # creating the transformation
                    tra2 = torch.nn.Sequential(transforms.Resize(HxW))
                    tmp_query = tra(original).to(args.device)  # transforming the img
                    img = tra2(tmp_query)
                    descriptors = model("features_extractor", [img, "global"])
                    multi_scale.append(descriptors)
                feature = torch.stack(multi_scale, -1)
                descriptors, min_index = torch.min(feature.type(torch.float32), dim=-1)
                del min_index
            else:
                images.to(args.device)
                descriptors = model("features_extractor", [images, "global"])

            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            images.to(args.device)
            descriptors = model("features_extractor", [images, "global"])
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]

    # Use a kNN to find predictions     ----    faiss (Facebook AI Similarity Search)

    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
                # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str, predictions


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def open_image(path):
    return Image.open(path).convert("RGB")


def use_rerank(model, predictions, test_dataset, num_reranked_predictions=5, test_batch_size=16):
    """Compute the test by warping the query-prediction pairs.

    Parameters
    ----------
    model : network.Network
    predictions : np.array of int, containing the first 20 predictions for each query, with shape [queries_num, 20].
    test_dataset : dataset_geoloc.GeolocDataset, which contains the test-time images (queries and gallery).
    num_reranked_predictions : int, how many predictions to re-rank.
    test_batch_size : int.

    Returns
    -------
    recalls : np.array of int, containing R@1, R@5, r@10, r@20.
    recalls_pretty_str : str, pretty-printed recalls
    """

    model = model.eval()
    reranked_predictions = predictions.copy()
    with torch.no_grad():
        for num_q in tqdm(range(test_dataset.queries_num), desc="Testing", ncols=100):

            dot_prods_wqp = np.zeros((num_reranked_predictions))
            query_path = test_dataset.queries_paths[num_q]

            for i1 in range(0, num_reranked_predictions, test_batch_size):

                batch_indexes = list(range(num_reranked_predictions))[i1:i1 + test_batch_size]
                current_batch_size = len(batch_indexes)
                pil_image = open_image(query_path)
                query = base_transform(pil_image)
                query_repeated_twice = torch.repeat_interleave(query.unsqueeze(0), current_batch_size, 0)

                preds = []
                for i in batch_indexes:
                    pred_path = test_dataset.database_paths[predictions[num_q, i]]
                    pil_image = open_image(pred_path)
                    query = base_transform(pil_image)
                    preds.append(query)
                preds = torch.stack(preds)

                warped_pair = compute_warping(model, query_repeated_twice.cuda(), preds.cuda())
                q_features = model("features_extractor", [warped_pair[0], "local"])
                p_features = model("features_extractor", [warped_pair[1], "local"])
                # Sum along all axes except for B. wqp stands for warped query-prediction
                dot_prod_wqp = (q_features * p_features).sum(list(range(1, len(p_features.shape)))).cpu().numpy()

                dot_prods_wqp[i1:i1 + test_batch_size] = dot_prod_wqp

            reranking_indexes = dot_prods_wqp.argsort()[::-1]
            reranked_predictions[num_q, :num_reranked_predictions] = predictions[num_q][reranking_indexes]

    ground_truths = test_dataset.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(reranked_predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], ground_truths[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / test_dataset.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str