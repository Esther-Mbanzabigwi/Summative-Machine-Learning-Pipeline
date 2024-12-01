import pickle
import json
import torch


# Function for setting the states checkpoints
def save_checkpoint(path, model, optimizer, train_acc, train_loss, valid_acc, valid_loss, epoch, learning_rate, scheduler=None):
    """
        Function saves different objects states

        args:
            model: the model to be saved
            optimizer: The optimizer to be saved
            epoch -> int: The epoch to be saved
            learning_rate -> decimal: The learning rate to be save
            scheduler: the scheduler to be saved
    """

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "valid_acc": valid_acc,
        "valid_loss": valid_loss,
        "database": model.database
    }

    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, path) #Saving with .pth extension
    
    path = path.replace(".pth", ".pkl") # saving with pickle
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


# Function for loading checkpoint
def load_checkpoint(path, model, optimizer, scheduler=None, use_pickle=False, device='cpu'):
    checkpoint = None
    try:
        if use_pickle:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
    except:
        return None
    
    if checkpoint is None:
        return None

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    learning_rate = checkpoint["learning_rate"]
    train_acc = checkpoint["train_acc"]
    train_loss = checkpoint["train_loss"]
    valid_acc = checkpoint["valid_acc"]
    valid_loss = checkpoint["valid_loss"]
    
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        return model, optimizer, scheduler, learning_rate, epoch, train_acc, train_loss, valid_acc, valid_loss
    else:
        return model, optimizer, scheduler, learning_rate, epoch, train_acc, train_loss, valid_acc, valid_loss


# function which loads only the model
def load_model_checkpoint(path, model, use_pickle=False):
    checkpoint = None
    try:
        if use_pickle:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    if checkpoint is None:
        return None

    model.load_state_dict(checkpoint["model_state"])
    model.float()

    return model

def get_classes(path):
    classes = []
    with open(path, "r") as f:
        classes = json.load(f)
    return classes