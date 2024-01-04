from transformers import RobertaTokenizer, RobertaModel
import json
import jsonlines
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import Recommender
from config import *
from dataset import CustomDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

from tqdm import tqdm

def precision(y_true, y_pred):
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)
    res = precision_score(y_true_binary, y_pred_binary)
    return res

def recall(y_true, y_pred):
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)
    res = recall_score(y_true_binary, y_pred_binary)
    return res

def f1(y_true, y_pred):
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)
    res = f1_score(y_true_binary, y_pred_binary)
    return res

def mae(y_true, y_pred):
    y_pred = np.clip(y_pred, 1., 5.)
    res = mean_absolute_error(y_true, y_pred)
    return res

def rmse(y_true, y_pred):
    y_pred = np.clip(y_pred, 1., 5.)
    res = mean_squared_error(y_true, y_pred)
    return np.sqrt(res)

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
    state_dict=torch.load("heterogeneous-model/checkpoint/cp.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    valdataset = CustomDataset(anot[-10000:], rating_matrix, movies, config)
    valloader = DataLoader(dataset=valdataset, batch_size=24, shuffle=False, generator=torch.Generator(device="cpu"))

    l1_loss = nn.L1Loss()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch in tqdm(valloader):
            out = model(batch)
            y_pred += out.reshape(-1).cpu().tolist()
            y_true += batch["candidate_item"]["rating"].reshape(-1).cpu().tolist()
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(user_rating.rating.to_numpy().reshape(-1,1)/2)
    y_true = scaler.inverse_transform(np.array(y_true).reshape(1,-1))[0]
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(1,-1))[0]
    print(f"Precision: {precision(y_true,y_pred)}")
    print(f"Recall: {recall(y_true,y_pred)}")
    print(f"F1: {f1(y_true,y_pred)}")
    print(f"MAE: {mae(y_true,y_pred)}")
    print(f"RMSE: {rmse(y_true,y_pred)}")