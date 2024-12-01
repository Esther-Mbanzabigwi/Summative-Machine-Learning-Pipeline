import os
import gc
import json
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from .helpers import get_classes, save_checkpoint, load_checkpoint
from .config import CONFIG
from .model import Resnet50, Block

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print(f"Device: {device}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


## Some constant variables
CHECKPOINT_PATH = os.path.join("Summative-MLOP", "models")

KAGGLE_PATH = os.path.join("Summative-MLOP", "Data", "data")
KAGGLE_DOWN = "New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(KAGGLE_PATH, KAGGLE_DOWN, "train")
VALID_DIR = os.path.join(KAGGLE_PATH, KAGGLE_DOWN, "valid")
TEST_DIR = os.path.join(KAGGLE_PATH, "test")

# Class Label
TRAIN_LABELS = sorted(os.listdir(TRAIN_DIR))
VALID_LABELS = sorted(os.listdir(VALID_DIR))
TEST_LABELS = sorted(os.listdir(TEST_DIR))



# configurations

classes = get_classes(os.path.join("Data", "classes.json"))
CONFIG["num_classes"] = len(classes)

print(f"Configurations:\n")
for key, value in CONFIG.items():
  print(f"{key}: {value}\n")


data_mean = (0.485, 0.456, 0.406)
data_std = (0.229, 0.224, 0.225)

transformations = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std)
]
)  # converting to tensor and resizing all images to 256x256

train_data = ImageFolder(root=TRAIN_DIR, transform=transformations)
valid_data = ImageFolder(root=VALID_DIR, transform=transformations)
test_data = ImageFolder(root=TEST_DIR, transform=transformations)


# Data loaders
train_loader = DataLoader(train_data,
                         batch_size=CONFIG['batch_size'],
                         shuffle     = True,
                         num_workers = 2,
                         pin_memory  = True)


val_loader = DataLoader(valid_data,
                        batch_size=CONFIG['batch_size'],
                        shuffle     = False,
                        num_workers = 2,
                        pin_memory  = True)

test_loader = DataLoader(test_data,
                        batch_size=CONFIG['batch_size'],
                        shuffle     = False,
                        num_workers = 2,
                        pin_memory  = True)




gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=device, abbreviated=False)

# Tesing the model creation
for Xb, yb in train_loader:
  Xb, yb = Xb.to(device), yb.to(device)
  print(f"Images shape: {Xb.shape}")
  print(f"Labels shape: {yb.shape}")
  break

model = Resnet50(Block, Xb.shape[1], 62, 4, CONFIG["num_classes"]).to(device)


# defining optimizer
optimizer = optim.AdamW(model.parameters(),
                      lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])


# loss function
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])


# scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='max',
                                                 factor= CONFIG['scheduler_factor'],
                                                 patience=CONFIG['scheduler_patience'],
                                                 min_lr=CONFIG['min_lr'],
                                                threshold=CONFIG['scheduler_threshold'])

# Defining scaler
if torch.cuda.is_available():
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = torch.cuda.amp.GradScaler()


checkpoint_name = "best_checkpoint.pth"
checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_name)
experiments_logs_path = os.path.join(CHECKPOINT_PATH, 'Experiments')
os.makedirs(experiments_logs_path, exist_ok = True)

if torch.cuda.is_available():

    torch.cuda.empty_cache()

gc.collect()

best_valid_acc = 0

e = 0

epoch_time_ellapse = {}
RESUME_LOGGING = False

if RESUME_LOGGING:

    (model, optimizer, scheduler, curr_lr, e,

     train_acc, train_loss, best_valid_acc, valid_loss) = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)


    print("""Resuming from epoch {} with \ntrain acc: {:.04f}% val acc: {:.04f}% train loss: {:.04f}

    valid loss: {:.04f} lr: {:.04f}\n Optimizer: """.format(e, train_acc, best_valid_acc, train_loss, valid_loss, curr_lr, optimizer))


if torch.cuda.is_available():
    torch.cuda.empty_cache()

gc.collect()

if e == CONFIG["epochs"]:
    CONFIG["epochs"] += 5


with open(f'{experiments_logs_path}/experiment_log.txt', "w") as f:
    for epoch in range(e, CONFIG["epochs"] + 1):
        epoch_start_time = datetime.now()
        epoch_line = "\nEpoch {}/{}".format(epoch+1, CONFIG['epochs'])
        print(epoch_line)
        curr_lr = float(optimizer.param_groups[0]['lr'])

        train_acc, train_loss = model.train_model(optimizer, train_loader, criterion, scaler, device)
        val_acc, val_loss     = model.validate_model(val_loader, criterion, device)


        epoch_line += "\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc, train_loss, curr_lr)
        epoch_line += "\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc, val_loss)

        print(epoch_line)

        if scheduler is not None:
            scheduler.step(val_acc)


        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            save_checkpoint(checkpoint_path, model, optimizer, train_acc, train_loss, val_acc, val_loss, epoch, curr_lr, scheduler)

            print("\nSaved the best model\n")

            epoch_line += "\nSaved the best model\n"

        time_epoch_took = (datetime.now() - epoch_start_time).seconds

        print(f"Epoch {epoch + 1} took: {time_epoch_took // 60} minutes and {time_epoch_took % 60 } seconds")

        epoch_line += f"Epoch {epoch + 1} took: {time_epoch_took // 60} minutes and {time_epoch_took % 60 } seconds"


        epoch_time_ellapse[epoch] = time_epoch_took

        f.writelines(epoch_line)

    f.close()

with open(f'{experiments_logs_path}/epoch_time_ellapse.json', 'w') as f:

    json.dump(epoch_time_ellapse, f)

    f.close()

print(f"Best Validation Accuracy: {best_valid_acc}")