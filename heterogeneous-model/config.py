config = {
    "itemencoder":
        {"categorical": {
            "single": {"collection": {"vocab_size": 4483},
                       "crew": {"vocab_size": 17693}
                      },
            "multi": {"genre": {"max_seq_len": 8, "vocab_size": 20+1},
                      "company": {"max_seq_len": 5, "vocab_size": 23674+1},
                      "cast": {"max_seq_len": 3, "vocab_size": 47630+1}
                     },
            "d_model": 16
        },
        "text": {"max_seq_len": 256, "num_encoder": 2, "d_model": 768},
        "ff": {"dim": 512, "dropout": 0.2}},
    "userencoder": {"userid": {"vocab_size": 10000, "d_model": 512},
                    "rating": {"vocab_size": 10, "d_model": 512},
                    "n_en_item": 20,
                    "n_can_item": 10}
}

device="cuda"