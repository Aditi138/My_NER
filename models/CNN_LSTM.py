import torch
import torch.nn as nn
import torch.nn.modules.dropout
import torchwordemb
from torch.autograd import Variable
from utils import *



class CNN_LSTM(nn.Module):
    def __init__(self,ner_data_loader, args, padding=1,dropout=0):
        super(CNN_LSTM, self).__init__()
        self.embedding_dim = args.char_emb_dim
        self.filter_size = args.cnn_filter_size
        self.win_size = args.cnn_win_size
        self.padding = padding
        self.char_vocab_size = ner_data_loader.char_vocab_size
        self.dropout = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(self.char_vocab_size, self.embedding_dim)


        self.layer = nn.Conv1d(1 , self.filter_size, self.win_size, padding=self.padding, stride=1 , bias=True)
        self.tanhlayer = nn.Tanh()


    def forward(self, input):
        input = input.unsqueeze(0)
        embs_dropout = self.dropout(input)
        conv1 = self.layer(embs_dropout)
        tanh = self.tanhlayer(conv1)

        #Since variable length sequence
        maxPool = nn.MaxPool1d(input.size(2), stride =1)
        output = maxPool(tanh).view(self.filter_size)

        return output

    def encode(self, input_sents):
        sents_emb=[]
        for sent in  input_sents:
            sent_emb=[]
            input_embs = []
            for word in sent:
                if len(word) < self.win_size:
                    word = word + [0]*(self.win_size - len(word))
                for char in word:
                    input_embs.append(self.embeddings(Variable(torch.LongTensor([char]))))
                cat_embs = torch.cat(input_embs,0)
                word_emb = self.forward(cat_embs.transpose(0,1))
                sent_emb.append(word_emb)
            sents_emb.append(sent_emb)


        maxlen = max(len(sent) for sent in input_sents)
        padded_sent = []
        for i in range(maxlen):
            word_i = [sent[i] if i < len(sent_emb) else torch.zeros(self.embedding_dim) for sent_emb in sents_emb]
            word_i = torch.cat(word_i,1).view(self.embedding_dim,len(input_sents))
            padded_sent.append(word_i)

        return padded_sent

class LookpuEncoder(nn.Module):
    def __init__(self, embedding_dim,vocab_size =0, pretrain_path=None, dropout=0, padding_token = None,pretrain_embedding=None):
        super(LookpuEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.pretrain_path = pretrain_path
        self.padding_token = padding_token

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if pretrain_embedding is not None:
               #self.word_embeddings.weight = nn.Parameter(pretrain_embedding)
               self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))
          #self.word_to_id, self.word_embeddings  = torchwordemb.load_glove_text(pretrain_path)

    def encode(self, input_sents):
        sents_emb = []
        transpose_inpus,_ = transpose_input(input_sents, self.padding_token)
        transpose_inpus = torch.LongTensor(transpose_inpus)
        sents = transpose_inpus.transpose(0,1)
        for sent in sents:
            sent_emb = []
            for word in sent:
                sent_emb.append(self.word_embeddings(Variable(torch.LongTensor([word]))))
            sentence_embs = torch.cat(sent_emb, 0)
            sents_emb.append(sentence_embs)
            #sents_emb.append(word_emb)


        return sents_emb


class BiLSTM(nn.Module):
    def __init__(self,emb_dim, output_dropout_rate, emb_dropout_rate,hidden_dim,vocab_size=0,padding_token=None,layer=1):
        super(BiLSTM, self).__init__()
        self.embedding_dim = emb_dim
        self.input_dim = emb_dim
        self.vocab_size = vocab_size
        self.padding_token = padding_token
        self.drop_out_rate = output_dropout_rate
        self.emb_drop_rate = emb_dropout_rate
        self.hidden_dim = hidden_dim * 2
        self.dropout = nn.Dropout(self.emb_drop_rate)

        # self.fwd_rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=layer)
        # self.bwd_rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=layer)

        if vocab_size > 0:
            print "In BiRNN, creating lookup table!"
            self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        else:
            self.embeddings = None

        self.fwd_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1)
        self.bwd_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                                num_layers=1)


    def init_hidden(self):
        return (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def forward(self, input, input_r):
        seq_len  = input.size(0)
        features = input.size(1)

        #getting forward vectors
        input = input.view(seq_len, 1, features)
        lstm_out, hidden = self.fwd_lstm(input, self.init_hidden())
        lstm_out = lstm_out.view(seq_len, self.hidden_dim // 2)

        # getting backward vectors
        input_r = input_r.view(seq_len, 1, features)
        lstm_r, _ = self.bwd_lstm(input_r, self.init_hidden())
        lstm_r = lstm_r.view(seq_len, self.hidden_dim // 2)
        return lstm_out, lstm_r

    def encode(self, input_sents):
        sents_emb = []

        for sent in input_sents:
            sent_emb = []

            for word in sent:
                input_embs = []
                for char in word:
                    print char
                    input_embs.append(self.embeddings(Variable(torch.LongTensor([char]))))

                cat_embs = torch.cat(input_embs, 0)
                cat_embs_r = torch.cat(input_embs[::-1],0)

                word_emb, word_emb_r= self.forward(cat_embs,cat_embs_r)
                WE = torch.cat((word_emb[-1], word_emb_r[-1]), dim=0).view(1,self.hidden_dim)
                sent_emb.append(WE)

            sentence_emb = torch.cat(sent_emb,0)
            sents_emb.append(sentence_emb)
         # Maps the output of the LSTM into tag space.


        return sents_emb

class CRF(nn.Module):
    def __init__(self, ner_data_loader, args):
        super(CRF, self).__init__()
        self.tag_emb_dim = args.tag_emb_dim
        self.ner_vocab_size = ner_data_loader.ner_vocab_size
        self.hidden_dim  = args.char_hidden_dim
        tag_to_id  = ner_data_loader.tag_to_id
        self.constraints = [[[tag_to_id["B-GPE"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-ORG"]] * 3, [tag_to_id["I-GPE"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-PER"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-GPE"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-LOC"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-GPE"]]],
                            [[tag_to_id["O"]] * 4,
                             [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"], tag_to_id["I-GPE"]]],
                            [[tag_to_id["I-GPE"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-ORG"]] * 3, [tag_to_id["I-GPE"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-PER"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-GPE"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-LOC"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-GPE"]]]]

class BiLSTM_CRF(nn.Module):
    def __init__(self,ner_data_loader,args):
        super(BiLSTM_CRF,self).__init__()

        self.char_encoder = BiLSTM(emb_dim=args.char_emb_dim,
                                   vocab_size = ner_data_loader.ipa_char_vocab_size,
                                   output_dropout_rate = args.output_dropout_rate,
                                   emb_dropout_rate = args.emb_dropout_rate,
                                   hidden_dim = args.char_hidden_dim)

        if args.pretrain_emb_path is None:
            self.word_lookup = LookpuEncoder(embedding_dim=args.word_emb_dim,
                                             vocab_size=ner_data_loader.word_vocab_size,
                                             dropout=args.emb_dropout_rate,
                                             padding_token=ner_data_loader.word_padding_token)
        else:
            print "Using pretrained embeddings!"
            self.word_lookup = LookpuEncoder(embedding_dim=args.word_emb_dim,
                                             vocab_size=ner_data_loader.word_vocab_size,
                                             dropout=args.emb_dropout_rate,
                                             pretrain_path=args.pretrain_emb_path,
                                             padding_token=ner_data_loader.word_padding_token,
                                             pretrain_embedding = ner_data_loader.pretrain_word_emb)

        birnn_input_dim = args.char_hidden_dim * 2 + args.word_emb_dim
        self.word_encoder  = BiLSTM(emb_dim=birnn_input_dim,
                                    hidden_dim=args.hidden_dim,
                                    emb_dropout_rate=args.emb_dropout_rate,
                                    output_dropout_rate=args.output_dropout_rate)


    def encode(self, sents, char_sent, ipa_sent, ner_tags, training=True):
        char_embs = self.char_encoder.encode(ipa_sent)
        word_embs = self.word_lookup.encode(sents)

        char_word = [torch.cat((c,w),dim=1) for c,w in zip(char_embs, word_embs)]
        return word_embs



