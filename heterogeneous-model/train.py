from transformers import RobertaTokenizer, RobertaModel
import json
import jsonlines
import random
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import Recommender
from config import *
from dataset import CustomDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    movie_df = pd.read_csv("processed_data/movie_credits_processed.csv")
    user_rating = pd.read_csv("processed_data/link_ratings.csv")

    user_rating["rating"] = (user_rating["rating"]*2 - 1).astype(np.int32)
    user_rating = user_rating[user_rating["rating"] != -1]
    user_rating = user_rating.rename(columns={"movieId": "id"})

    movie_df["crew"] = movie_df["crew"].apply(lambda x: json.loads(x)[0])
    user_rating = user_rating[["userId","tmdbId","rating"]]
    movie_df = movie_df[["id", "belongs_to_collection", "crew", "genres", "production_companies", "cast", "concatenated"]].drop_duplicates()

    genres = []
    genres_mask = []
    company = []
    company_mask = []
    cast = []
    cast_mask = []

    def pad_trunc_mask(value, feature):
        f_cf = config["itemencoder"]["categorical"]["multi"][feature]
        value = json.loads(value)[:f_cf["max_seq_len"]]
        if len(value) != 0:
            _max, _min = max(value), min(value)
        else:
            _max, _min = 0, 0
        att_mask = [1]*len(value)
        if len(value) < f_cf["max_seq_len"]:
            att_mask += (f_cf["max_seq_len"] - len(value))*[0]
            value += (f_cf["max_seq_len"] - len(value))*[f_cf["vocab_size"]-1]
        att_mask[0]=1
        return value, att_mask, _max, _min

    max_g, max_c, max_ca = 0, 0, 0
    min_g, min_c, min_ca = 1e7, 1e7, 1e7

    for i in range(len(movie_df)):
        row = movie_df.iloc[i]
        g_v, g_mask, g, _g = pad_trunc_mask(row.genres, "genre")
        if g > max_g: max_g = g
        if _g < min_g: min_g = _g
        genres.append(g_v)
        genres_mask.append(g_mask)
        c_v, c_mask, c, _c = pad_trunc_mask(row.production_companies, "company")
        if c > max_c: max_c = c
        if _c < min_c: min_c = _c
        company.append(c_v)
        company_mask.append(c_mask)
        ca_v, ca_mask, ca, _ca = pad_trunc_mask(row.cast, "cast")
        if ca > max_ca: max_ca = ca
        if _ca < min_ca: min_ca = _ca
        cast.append(ca_v)
        cast_mask.append(ca_mask)

    movie_df["genres"] = genres
    movie_df["genres_mask"] = genres_mask
    movie_df["production_companies"] = company
    movie_df["company_mask"] = company_mask
    movie_df["cast"] = cast
    movie_df["cast_mask"] = cast_mask
    print(max_g, max_c, max_ca)
    print(min_g, min_c, min_ca)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    text, text_mask = [], []
    for i in range(len(movie_df)):
        ids = tokenizer(movie_df.iloc[i].concatenated, padding='max_length', truncation=True, max_length=config["itemencoder"]["text"]["max_seq_len"])
        text.append(ids["input_ids"])
        text_mask.append(ids["attention_mask"])
    movie_df["text"] = text
    movie_df["text_mask"] = text_mask

    scaler = MinMaxScaler(feature_range=(-1, 1))
    user_rating["scaled_rating"] = scaler.fit_transform(user_rating.rating.to_numpy().reshape(-1,1)).reshape(len(user_rating),)

    with jsonlines.open('processed_data/samples.jsonl', 'r') as jsonl_f:
        anot = [obj for obj in jsonl_f]
    random.seed(20)
    random.shuffle(anot)

    movies = movie_df.set_index("id").to_dict(orient='index')

    rating_matrix = {}
    for i in range(len(user_rating)):
        rating_matrix[user_rating.iloc[i].userId] = {}
    for i in range(len(user_rating)):
        row = user_rating.iloc[i]
        rating_matrix[row.userId][row.tmdbId] = {"rating": row.rating, "scaled_rating": row.scaled_rating}

    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    model = Recommender(RobertaModel.from_pretrained('roberta-base'), config)
    model.to(device)

    traindataset = CustomDataset(anot[:-15000], rating_matrix, movies, config)
    trainloader = DataLoader(dataset=traindataset, batch_size=8, shuffle=True, generator=torch.Generator(device="cpu"))
    valdataset = CustomDataset(anot[-15000:-10000], rating_matrix, movies, config)
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
                    torch.save(model.state_dict(), "heterogeneous-model/checkpoint/cp.pt")
    #             wandb.log({"train_loss": train_loss/len(trainloader), "val_loss": val_loss/len(valloader)})
                print(f"val loss: {val_loss/len(valloader)}")