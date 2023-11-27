import torch
from tqdm.notebook import tqdm

def TrainLoopV1(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    train_loader:torch.utils.data.DataLoader,
    val_loader:torch.utils.data.DataLoader,
    scheduler=None,
    n_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    batch_loss:int=1,
    device:str='cuda'
):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_weights = model.state_dict()
    for epoch in tqdm(range(n_epochs)):
        model.train()
        print("\n---------------------\nEpoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, images, target in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target]
            optimizer.zero_grad()
            loss_dict = model(images, target)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses
            losses.backward()
            optimizer.zero_grad()
            if i % batch_loss == 0:
                print("Loss for Batch {} = {}".format(i, losses))

        print("Loss for epoch {} = {}".format(epoch, train_loss))

        model.eval()
        with torch.inference_mode():
            validation_loss = 0
            for images, target in val_loader:
                images = list(image.to(device) for image in images)
                target = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target]
                loss_dict = model(images, target)
                losses = sum(loss for loss in loss_dict.values())
                validation_loss += losses
            
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")
        if scheduler is not None:
            try:
                scheduler.step(validation_loss)
            except:
                scheduler.step()
        if epochs_without_improvement == early_stopping_rounds:
            break

    if return_best_model == True:
        model.load_state_dict(best_weights)