from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import Recommender
from config import *
from dataset import CustomDataset

from tqdm import tqdm

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = Recommender(RobertaModel.from_pretrained('roberta-base'), config)
    state_dict=torch.load("/kaggle/input/movie-dataset-lllll/cp.pt", map_location=device)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(new_state_dict)
    model.to(device)

    traindataset = CustomDataset(anot[:-5000], rating_matrix, movies, config)
    trainloader = DataLoader(dataset=traindataset, batch_size=8, shuffle=True, generator=torch.Generator(device="cpu"))
    valdataset = CustomDataset(anot[-10000:], rating_matrix, movies, config)
    valloader = DataLoader(dataset=valdataset, batch_size=24, shuffle=False, generator=torch.Generator(device="cpu"))

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.00005}], weight_decay=1e-5)

    n_epoch=10
    print("Start .....")
    best_loss = 1e6
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    for epoch in range(n_epoch):
        print(f"epoch {epoch} ----------------------------")
        model.train()
        train_loss = 0

        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            out = model(batch)
            loss = mse_loss(out, batch["candidate_item"]["rating"])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)

            optimizer.step()

            train_loss += loss.item()
    #         except:
    #             print(batch["idx"])

        else:
            print(f"train loss: {train_loss/len(trainloader)}")
            # torch.save(model.state_dict(), "source/weights/last_vgg_transformer.pt")
            with torch.no_grad():
                model.eval()
                val_loss = 0
                for batch in tqdm(valloader):
                    out = model(batch)
                    loss = l1_loss(out, batch["candidate_item"]["rating"])
                    val_loss += loss
                if best_loss > val_loss/len(valloader):
                    best_loss = val_loss/len(valloader)
                    torch.save(model.state_dict(), "/kaggle/working/cp.pt")
    #             wandb.log({"train_loss": train_loss/len(trainloader), "val_loss": val_loss/len(valloader)})
                print(f"val loss: {val_loss/len(valloader)}")