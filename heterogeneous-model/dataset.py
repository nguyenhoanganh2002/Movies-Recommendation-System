from torch.utils.data import Dataset
from config import *
import torch
from transformers import RobertaTokenizer

class CustomDataset(Dataset):
    def __init__(self, anot, rating_matrix, movies, config):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.anot = anot
        self.rating_matrix = rating_matrix
        self.movies = movies
        self.max_seq_len = config["itemencoder"]["text"]["max_seq_len"]

    def __len__(self):
        return len(self.anot)
    
    def get_en(self, lsid, userId):
        en_item = {
            "userid": [userId],
            "rating": [],
            "text": [],
            "text_mask": [],
            "col": [],
            "crew": [],
            "gen": [],
            "gen_mask": [],
            "com": [],
            "com_mask": [],
            "cast": [],
            "cast_mask": []
        }
        for id in lsid:
            try:
                movie = self.movies[id]
            except:
#                 print(id)
                movie = self.movies[89]
            try:
                rate = self.rating_matrix[userId][id]
            except:
#                 print(userId, id)
                rate = {"rating": 4}
            en_item["rating"].append(rate["rating"])
            en_item["text"].append(movie["text"])
            en_item["text_mask"].append(movie["text_mask"])
            en_item["col"].append(movie["belongs_to_collection"])
            en_item["crew"].append(movie["crew"])
            en_item["gen"].append(movie["genres"])
            en_item["gen_mask"].append(movie["genres_mask"])
            en_item["com"].append(movie["production_companies"])
            en_item["com_mask"].append(movie["company_mask"])
            en_item["cast"].append(movie["cast"])
            en_item["cast_mask"].append(movie["cast_mask"])
        
        for key in en_item.keys():
            en_item[key] = torch.LongTensor(en_item[key]).to(device)
        
        return en_item
    
    def get_candidate(self, lsid, userId):
        candidate_item = {
            "text": [],
            "text_mask": [],
            "col": [],
            "crew": [],
            "gen": [],
            "gen_mask": [],
            "com": [],
            "com_mask": [],
            "cast": [],
            "cast_mask": []
        }
        rating = []
        for id in lsid:
            try:
                movie = self.movies[id]
            except:
#                 print(id)
                movie = self.movies[89]
            try:
                rate = self.rating_matrix[userId][id]
            except:
#                 print(userId, id)
                rate = {"scaled_rating": 4}
            rating.append(rate["scaled_rating"])
            candidate_item["text"].append(movie["text"])
            candidate_item["text_mask"].append(movie["text_mask"])
            candidate_item["col"].append(movie["belongs_to_collection"])
            candidate_item["crew"].append(movie["crew"])
            candidate_item["gen"].append(movie["genres"])
            candidate_item["gen_mask"].append(movie["genres_mask"])
            candidate_item["com"].append(movie["production_companies"])
            candidate_item["com_mask"].append(movie["company_mask"])
            candidate_item["cast"].append(movie["cast"])
            candidate_item["cast_mask"].append(movie["cast_mask"])
        
        for key in candidate_item.keys():
            candidate_item[key] = torch.LongTensor(candidate_item[key]).to(device)
        candidate_item["rating"] = torch.FloatTensor(rating).to(device)
        return candidate_item
        

    def __getitem__(self, idx):
        sample = self.anot[idx]
        return {
            "en_item": self.get_en(sample["encode_id"], sample["userId"]),
            "candidate_item": self.get_candidate(sample["rate_id"], sample["userId"])
        }