import argparse
from dataloaders.data_loader import *
from models.CNN_LSTM import *
from models.utils import  *
import torch


def main(args):
    ner_data_loader = NER_DataLoader(args)
    print ner_data_loader.id_to_tag

    sents, char_sents, tgt_tags, discrete_features,bc_features, ipa_char_sents = ner_data_loader.get_data_set(args.train_path, args.lang)

    print ner_data_loader.ipa_char_to_id
    #model = CNN_LSTM(ner_data_loader, args)
    #model = BiLSTM(ner_data_loader, args)
    model = BiLSTM_CRF(ner_data_loader, args)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    total_loss = torch.Tensor([0])

    for epoch in range(10):
         for b_sents, b_char_sents, b_ner_tags, b_ipa in make_bucket_batches(zip(sents, char_sents, tgt_tags,ipa_char_sents), args.batch_size):
             model.zero_grad()
             #loss = model.encode(b_ipa)
             loss = model.encode(b_sents, b_char_sents, b_ipa, b_ner_tags)








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=5783287, type=int)

    parser.add_argument("--lang", default="english", help="the target language")
    parser.add_argument("--train_path", default="../datasets/english/eng.train.bio.conll", type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)

    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=50, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--char_hidden_dim", default=25, type=int)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)

    parser.add_argument("--output_dropout_rate", default=0.5, type=float)
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float)
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)

    parser.add_argument("--use_discrete_features", default=False, action="store_true")
    parser.add_argument("--use_ipa", default=False, action="store_true")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--isLr", default=False, action="store_true")
    parser.add_argument("--feature_dim", type=int, default=30)


    args = parser.parse_args()
    main(args)