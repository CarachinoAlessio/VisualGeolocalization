from datasets.test_dataset import TestDataset
import test
from model import network
import torch
from datetime import datetime
import parser

#Code inspired from https://github.com/mlfoundations/model-soups/blob/main/main.py

def load_model(model_path,args):
    #Creating same model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(delete_discriminator_layer(model_state_dict))
    return model

def greedy_soup(models_list, val_folder, args):
    sorted_models = []
    val_ds = TestDataset(val_folder,positive_dist_threshold=args.positive_dist_threshold, queries_folder="queries_v1",args=args)
    
    for (model,val) in models_list:
    
        model = model.to(args.device)
        model = model.eval()
        sorted_models.append((model, val))
        #continue
        recalls, recalls_str = test.test(args, val_ds, model)
        print(f"recalls: ", recalls)

    sorted_models.sort(key=compare, reverse=True)
    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = sorted_models[0][0].state_dict()
    num_ingredients = len(greedy_soup_ingredients)
    best_val_rec = sorted_models[0][1]

    for i in range(1,len(sorted_models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }
        new_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
        new_model.load_state_dict(potential_greedy_soup_params)
        new_model = new_model.to(args.device)
        new_model = new_model.eval()
        new_recall, _ = test.test(args, val_ds, new_model)
        print(f'Potential greedy soup val acc {new_recall[0]}, best so far {best_val_rec}.')
        if new_recall[0] > best_val_rec:
            greedy_soup_ingredients.append(sorted_models[i][0])
            best_val_rec = new_recall[0]
            greedy_soup_params = potential_greedy_soup_params
            print(f'Adding to soup.')
    experiment_name= "sample_soup"
    torch.save(greedy_soup_params, f"{experiment_name}.pth")


def compare(m1):
    return m1[1]

def delete_discriminator_layer(model):
    if "discriminator.1.weight" in model:
            del model["discriminator.1.weight"]
            del model["discriminator.1.bias"]
            del model["discriminator.3.weight"]
            del model["discriminator.3.bias"]
            del model["discriminator.5.weight"]
            del model["discriminator.5.bias"]
    return model

if __name__ == "__main__":

    
    parser = parser.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/content/sf-xs",
        help="folder to the dataset, if not specified, then is sf-xs",
    )
    args = parser.parse_arguments(is_training=False)
    models_path=["/content/logs/content/logs/default/trained_with_arcface/best_model.pth", #arcfare
                        "/content/logs/content/logs/default/trained_with_cosface/best_model.pth", #cosface
                        "/content/logs/content/logs/default/trained_with_sphereface/best_model.pth"] #sphereface
    val_rec = [1.0, 1.2, 1.2, 1.0, 1.2, 1.1, 1.1, 1.2, 1.0]
    models = []
    for index, model_path in enumerate(models_path):
        m = load_model(format(model_path),args)
        models.append((m, val_rec[index]))
    greedy_soup(models, args.dataset+"/test/", args)