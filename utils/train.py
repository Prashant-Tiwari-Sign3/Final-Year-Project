import torch
from tqdm.notebook import tqdm

def TrainLoopV1(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
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
        print("Epoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, image, mask in enumerate(train_dataloader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, mask)
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                print("Loss for batch {} = {}".format(i, loss))
        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))

        model.eval()
        validation_loss = 0
        with torch.inference_mode:
            for image, mask in val_dataloader:
                image, mask = image.to(device), mask.to(device)
                outputs = model(image)
                loss = criterion(outputs, mask)
                validation_loss += loss
            
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