import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, input_type="single"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_type = input_type
        if self.input_type == "multi":
            self.W = nn.Linear(d_model, 1)
            self.ff = nn.Linear(d_model, d_model)
        else:
            self.ff = nn.Linear(d_model, d_model)
        self.ff_dropout = nn.Dropout(0.1)
        self.ff_out = nn.Linear(d_model, d_model)
    
    def forward(self, input_ids, att_mask=None):
        embedded = self.embedding(input_ids) # B*L, N, d_modelS
        if self.input_type == "multi":
            att = self.W(embedded).reshape(att_mask.size())
            if att_mask is not None:
                att = att.masked_fill_(att_mask==0, -float('inf'))
            att_weights = F.softmax(att, dim=-1)
            hid = torch.mul(self.ff(embedded), att_weights.unsqueeze(-1)).sum(dim=-2)
#             print(hid.shape)
            return F.gelu(self.ff_dropout(self.ff_out(hid)))
        else:
            hid = F.gelu(self.ff(embedded))
#             print(hid.shape)
            return F.gelu(self.ff_dropout(self.ff(embedded)))[:,0,:]
        
    
class ItemEncoder(nn.Module):
    def __init__(self, roberta, config):
        super().__init__()
        # text encoder
        self.text_embedding = roberta.embeddings
        self.text_encoder = roberta.encoder.layer[:config["text"]["num_encoder"]]
        self.get_extended_attention_mask = roberta.get_extended_attention_mask
        
        # categorical encoders
        cat_d_model = config["categorical"]["d_model"]
        scat_cf = config["categorical"]["single"]
        self.collection_encoder = CategoricalEncoder(scat_cf["collection"]["vocab_size"], cat_d_model, input_type="single")
        self.crew_encoder = CategoricalEncoder(scat_cf["crew"]["vocab_size"], cat_d_model, input_type="single")
        
        mcat_cf = config["categorical"]["multi"]
        self.genre_encoder = CategoricalEncoder(mcat_cf["genre"]["vocab_size"], cat_d_model, input_type="multi")
        self.company_encoder = CategoricalEncoder(mcat_cf["company"]["vocab_size"], cat_d_model, input_type="multi")
        self.cast_encoder = CategoricalEncoder(mcat_cf["cast"]["vocab_size"], cat_d_model, input_type="multi")
        
        # feed forward
        en_out_dim = cat_d_model*5 + config["text"]["d_model"]
        self.ff = nn.Linear(en_out_dim, config["ff"]["dim"])
        self.ff_dropout = nn.Dropout(config["ff"]["dropout"])
        
#     def forward(self, col, lan, rel, crew, adult, gen, gen_mask, com, com_mask, cast, cast_mask, bud, text, text_mask):
    def forward(self, batch):
        """
        col, crew: B*Lx1
        gen, gen_mask, com, com_mask, cast, cast_mask: B*LxN_i
        text, text_mask: B*LxN_j
        """
        B, L = batch["col"].size()
        o_col = self.collection_encoder(batch["col"].reshape(B*L, -1))
        o_crew = self.crew_encoder(batch["crew"].reshape(B*L, -1))
        o_gen = self.genre_encoder(batch["gen"].reshape(B*L, -1), batch["gen_mask"].reshape(B*L, -1))
        o_com = self.company_encoder(batch["com"].reshape(B*L, -1), batch["com_mask"].reshape(B*L, -1))
        o_cast = self.cast_encoder(batch["cast"].reshape(B*L, -1), batch["cast_mask"].reshape(B*L, -1))
        o_text = self.forward_text_encoder(batch["text"].reshape(B*L, -1), batch["text_mask"].reshape(B*L, -1))
        hidden = self.ff_dropout(self.ff(torch.cat([o_col, o_crew, o_gen, o_com, o_cast, o_text], dim=-1)))
        return F.gelu(hidden)
        
    def forward_text_encoder(self, text_ids, att_mask):
        max_seq_length = att_mask.sum(dim=-1).max().item()
        text_ids = text_ids[:, :max_seq_length]
        att_mask = att_mask[:, :max_seq_length]
#         print(text_ids.size())
        batch_size, seq_length = text_ids.size()
        
        device = text_ids.device
        if hasattr(self.text_embedding, "token_type_ids"):
            buffered_token_type_ids = self.text_embedding.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(text_ids.size(), dtype=torch.long, device=device)
        hidden_state = self.text_embedding(input_ids=text_ids, token_type_ids=token_type_ids)
        
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(att_mask, text_ids.size())
        
        for layer in self.text_encoder:
            hidden_state = layer(hidden_state, extended_attention_mask)[0]
        return hidden_state[:, 0, :]

class UserEncoder(nn.Module):
    def __init__(self, roberta, config):
        super().__init__()
        self.item_encoder = ItemEncoder(roberta, config["itemencoder"])
        u_cf = config["userencoder"]
        d_model = u_cf["userid"]["d_model"]
        self.userID_embedding = nn.Embedding(u_cf["userid"]["vocab_size"], u_cf["userid"]["d_model"])
        self.rating_embedding = nn.Embedding(u_cf["rating"]["vocab_size"], u_cf["rating"]["d_model"])
        self.n_en_item = u_cf["n_en_item"]
        self.Wu = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.project = nn.Linear(d_model, d_model)
        
    def forward(self, batch):
        """
        rating: B*Lx1
        userid: Bx1
        """
        B, L = batch["rating"].size()
        item = self.item_encoder(batch) # B*L, d_model
#         item = item + self.rating_embedding(batch["rating"].reshape(B*L, -1)).squeeze(1)
#         item = item + self.userID_embedding(batch["userid"])
        B_L, d_model = item.size()
        item = item.reshape(B_L//self.n_en_item, self.n_en_item, d_model) # B, L, d_model
        item = item + self.userID_embedding(batch["userid"]).expand(B_L//self.n_en_item, self.n_en_item, d_model)
        embedded_id = self.rating_embedding(batch["rating"]).reshape(B, L, -1) # B, 1, d_model
#         print(item.shape)
#         print(embedded_id.shape)
#         print(B, L)
#         embedded_id = embedded_id.repeat(1,n_en_item).reshape(B_L//self.n_en_item, n_en_item, d_model) # B, L, d_model
        attn = F.tanh((self.Wu(embedded_id)*item).sum(dim=-1)).unsqueeze(1) # B, 1, L
#         print(attn.shape)
        attn_weights = F.softmax(attn, dim=-1).transpose(-1,-2)
        hidden = torch.mul(self.Wo(item), attn_weights).sum(dim=-2) # B, d_model
        hidden = self.project(hidden)
        return F.gelu(self.dropout(hidden))


class Recommender(nn.Module):
    def __init__(self, roberta, config):
        super().__init__()
        self.item_encoder = ItemEncoder(roberta, config["itemencoder"])
        self.user_encoder = UserEncoder(roberta, config)
        self.Q = nn.Linear(config["userencoder"]["userid"]["d_model"], config["userencoder"]["userid"]["d_model"])
        self.n_can_item = config["userencoder"]["n_can_item"]
    
    def forward(self, batch):
#         print("here-1")
        user = self.user_encoder(batch["en_item"]).unsqueeze(1) # B, 1, d_model
#         print("here-2")
        candidate_items = self.item_encoder(batch["candidate_item"]) # B, _L, d_model
        B_L, d_model = candidate_items.size()
        candidate_items = candidate_items.reshape(B_L//self.n_can_item, self.n_can_item, d_model)
#         print(user.shape)
#         print(candidate_items.shape)
        out = self.Q(user)@(candidate_items.transpose(-1,-2)) # B, 1, _L
        return F.tanh(out.squeeze(1))