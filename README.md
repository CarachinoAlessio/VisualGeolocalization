# Rethinking Visual Geo-localization for Large-Scale Applications

## Disclaimer
This work is still unfinished. You can check the current version of the report [here](./temp_REPORT.pdf).
Please note that the code in this repo is still not updated with respect to the results collected in the report.

## The project
This work aims to implement additional features to
the recent paper ”Rethinking Visual Geo-localization for
Large-Scale Applications” of CVPR(2022). 

Visual Geo-
localization consists in estimating the position where a
given photo, called query, was taken by comparing it with
a large database of images of known locations.

# Instructions to run and test the code on Colab😄

## Introduction
First of all, you should know that you can access to **two** Jupyter notebooks:
- The **first notebook** contains all the logs and tests we performed; it is accessible [here]().
- The **second notebook** contains the initial scripts you should run if you want to perform your own tests; it is accessible [here](./setup_for_tests.ipynb).

## Downloadable elements
The second notebook is ready-to-use and, by default, it downloads all datasets and models we used for our tests.

If you want to perform aimed tests with specific dataset, here is presented a table where, for each downloadable element, it is provided a brief description.

| **Element**    | **Description**                                                                                        |
|----------------|--------------------------------------------------------------------------------------------------------|
| sf_xs          | Dataset with San Francisco images                                                                      |
| tokyo_night    | Dataset with Tokyo images. It contains only night images                                               |
| tokyo_xs       | Dataset with Tokyo images                                                                              |
| night_target   | Dataset to perform data adaptation on night domain                                                     |
| logs           | It contains 4 saved models: (Cosface, Sphereface, Arcface, GRL), all trained with ResNet18 as backbone |
| geowarp_model  | It is a model that has been trained with ResNet18 as backbone and implements GeoWarp                   |
| eff2vs         | It is a model that has been trained with EfficientNetV2s as backbone                                   |
| eff2vs_geowarp | It is a model that has been trained with EfficientNetV2s as backbone and that implements GeoWarp       |
| eff2vs_grl     | It is a model that has been trained with EfficientNetV2s as backbone and that combines GeoWarp and GRL |

## Examples of commands
After you run all the scripts provided in the second notebook, you can use the following commands to run your experiments.
Of course, they are just an overview, if you want to see some more example you can access to the first notebook and see.

### Train CosPlace with CosFace/SphereFace/ArcFace
```!python train.py --dataset_folder /content/small --groups_num 1 --epochs_num 3 --loss_function=[select one among cosface, sphereface, arcface]```

### Test CosPlace with CosFace/SphereFace/ArcFace
```!python eval.py --dataset_folder /content/tokyo_xs/  --resume_model /content/content/logs/default/[select one among trained_with_cosface, trained_with_sphereface, trained_with_arcface]/best_model.pth```

### Train model with GRL for domain adaptation (and with EfficientNetV2s as backbone)
```!python train.py --dataset_folder /content/small --groups_num 1 --backbone efficientnet_v2_s --grl_param 0.3 --source_dir /content/small --target_dir /content/night_target```

### Test GRL model
```!python eval.py --dataset_folder /content/tokyo-night/  --resume_model /content/logs/content/logs/default/cosplace_with_grl/best_model.pth --grl_param 0.3```


### Test First type of Data Augmentation
```!python eval.py --dataset_folder /content/tokyo-night/  --resume_model /content/logs/content/logs/default/cosplace_with_grl/best_model.pth --grl_param 0.3 --night_test True --night_brightness 0.2```

### Train GeoWarp model on sf-xs and ResNet18 as backbone
```!python /content/train_geowarp.py --dataset_folder /content/small --groups_num 1 --epochs_num 3 ```

### Test GeoWarp on sf-xs
```!python /content/evalGeowarp.py --dataset_folder /content/small/ --resume_model /content/geowarp_model/best_model.pth ```

### Test with multiscale
```!python /content/eval.py --dataset_folder /content/tokyo-night --multi_scale --multi_scale_method=avg --select_resolution 0.526 0.588 1 1.7 1.9 --resume_model /content/logs/content/logs/default/cosplace_with_grl/best_model.pth --grl_param 0.3 ```

### Test ensembler (GeoWarp + GRL have been trained with EfficientNetV2s)
```!python eval_ensemble.py --dataset_folder /content/small/  --backbone efficientnet_v2_s --grl_param 0.3 --grl_model_path /content/eff2vs_grl/eff2vs_grl.pth --geowarp_model_path /content/eff2vs_geowarp/best_model.pth ```