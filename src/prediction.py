
import os
import torch
from model import Block, Resnet50
from helpers import load_model_checkpoint, get_classes
from config import CONFIG
from preprocessing import read_image


def make_prediction(model_path, image_path, classes_file_path, use_pickle=False, image=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    classes = get_classes(classes_file_path)
    CONFIG["num_classes"] = len(classes)

    print("Constructing the model")
    model = Resnet50(Block, 3, 62, 4, CONFIG["num_classes"]).to(device)

    print("Loadding model checkpoint")
    model = load_model_checkpoint(path=model_path, model=model, use_pickle=use_pickle)

    if model is None:
        print("Unable to load the model")
        return []
    
    print("Reading the image")
    dataloader = read_image(image_path=image_path, image=image)

    print("Making prediction")
    preds, predict_labels = model.predict(dataloader=dataloader, device=device, 
                                         labels_names=classes)
    
    print("Returning predicted disease")
    return predict_labels
    
if __name__ == "__main__":
    ROOT_PATH = os.path.abspath(__file__)
    ROOT_PATH = os.path.dirname(os.path.dirname(ROOT_PATH))
    # model_path = os.path.join(ROOT_PATH, "models", "best_checkpoint.pth")
    model_path = os.path.join(ROOT_PATH, "models", "best_checkpoint.pkl")
    image_path = os.path.join(ROOT_PATH, "Data", "data", "test", "test", "AppleCedarRust1.JPG")
    classes_file_path = os.path.join(ROOT_PATH, "Data","classes.json")
    use_pickle = True

    predictions = make_prediction(model_path=model_path, 
                                  image_path=image_path,
                                  classes_file_path=classes_file_path, 
                                  use_pickle=use_pickle,
                                  image=None)
    
    if len(predictions) > 0:
        print(f"Predicted disease: {predictions[-1]}")
    else:
        print("No predictions found")