import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import wandb
# Data augmentation and normalization for training
# Just normalization for validation
best_model_params_path = os.path.join('C:/Users/Kuzlik/PycharmProjects/DiBas', 'best_model_params.pt')
def train_model(model, dataloaders, device, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            wandb.log({f"{phase}_accuracy": epoch_acc, f"{phase}_loss": epoch_loss}, step=epoch)
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model
from sklearn.metrics import classification_report
def test_model(model, dataloaders, device):
    model.load_state_dict(torch.load(best_model_params_path))
    model.eval()  # Переключаем модель в режим оценки
    all_labels = []
    all_preds = []
    # Проходим по тестовому даталоадеру
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    # Выводим отчет о классификации
    report = classification_report(all_labels, all_preds, digits=4)
    print("Classification Report:")
    print(report)
    return report
if __name__ == '__main__':
    wandb.login(key="d068c3b6d7ec525f4ebce995148977b0d2c22da9")
    print(torch.cuda.is_available())
    num_epoch = 50
    lr = 0.001
    momentum = 0.9
    step_size = 7
    gamma = 0.1
    net = 'Resnet18'
    wandb.init(
        project="DIBAS21",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": net,
            "epochs": num_epoch,
            "momentum": momentum,
            "step_size": step_size,
            "gamma": gamma,
        }
    )
    from multiprocessing import freeze_support
    freeze_support()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Grayscale(3),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(3),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'C:/Users/Kirill/PycharmProjects/pythonProject4/data_test_tiff2'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=True, num_workers=1)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(weights=None)
    # model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)
    model_ft = train_model(model_ft, dataloaders, device, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=num_epoch)
    test_model(model_ft, dataloaders, device)

    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in dataloaders['test']:
        output = model_ft(inputs.to(device))  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(16, 11))
    sn.heatmap(df_cm, annot=True, linewidth=0.3, fmt='.2f', annot_kws={"size": 8})
    # sn.heatmap(df_cm, annot=True, linewidth=0.1, fmt='.2f')
    plt.savefig('output.png')
    wandb.log({"CM": wandb.Image("output.png")})
    wandb.finish()
    plt.figure(figsize=(22, 20))
