import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import wget
import os


def words2indices(origin, vocab):
    """ Transform a sentence or a list of sentences from str to int
    Args:
        origin: a sentence of type list[str], or a list of sentences of type list[list[str]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with int
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result


def indices2words(origin, vocab):
    """ Transform a sentence or a list of sentences from int to str
    Args:
        origin: a sentence of type list[int], or a list of sentences of type list[list[int]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with str
    """
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result


def pad(data, padded_token, device):
    """ pad data so that each sentence has the same length as the longest sentence
    Args:
        data: list of sentences, List[List[word]]
        padded_token: padded token
        device: device to store data
    Returns:
        padded_data: padded data, a tensor of shape (max_len, b)
        lengths: lengths of batches, a list of length b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


from itertools import chain
from collections import Counter
import json


class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]

    @staticmethod
    def build(data, max_dict_size, freq_cutoff, is_tags):
        """ Build vocab from the given data
        Args:
            data (List[List[str]]): List of sentences, each sentence is a list of str
            max_dict_size (int): The maximum size of dict
                                 If the number of valid words exceeds dict_size, only the most frequently-occurred
                                 max_dict_size words will be kept.
            freq_cutoff (int): If a word occurs less than freq_size times, it will be dropped.
            is_tags (bool): whether this Vocab is for tags
        Returns:
            vocab: The Vocab instance generated from the given data
        """
        word_counts = Counter(chain(*data))
        valid_words = [w for w, d in word_counts.items() if d >= freq_cutoff]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[: max_dict_size]
        valid_words += ['<PAD>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        if not is_tags:
            word2id['<UNK>'] = len(word2id)
            valid_words += ['<UNK>']
        return Vocab(word2id=word2id, id2word=valid_words)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf8') as f:
            entry = json.load(f)
        return Vocab(word2id=entry['word2id'], id2word=entry['id2word'])


class BiLSTMCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embed_size=256, hidden_size=256):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
            hidden_size (int): hidden state size
        """
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embedding = nn.Linear(312, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  # shape: (K, K)

    def forward(self, sentences, mask, tags, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            tags (tensor): corresponding tags, shape (b, len)
            sen_lengths (list): sentence lengths
        Returns:
            loss (tensor): loss on the batch, shape (b,)
        """
        sentences = sentences  # shape: (len, b)
        sentences = self.embedding(sentences).transpose(0, 1)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        loss = self.cal_loss(tags, mask, emit_score.transpose(0, 1))  # shape: (b,)
        return loss

    def encode(self, sentences, sent_lengths):
        """ BiLSTM Encoder
        Args:
            sentences (tensor): sentences with word embeddings, shape (len, b, e)
            sent_lengths (list): sentence lengths
        Returns:
            emit_score (tensor): emit score, shape (b, len, K)
        """
        hidden_states, _ = self.encoder(sentences)
        # hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def cal_loss(self, tags, mask, emit_score):
        """ Calculate CRF loss
        Args:
            tags (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len), values in PAD position is 0
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, mask, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            sen_lengths (list): sentence lengths
        Returns:
            tags (list[list[str]]): predicted tags for the batch
        """
        batch_size = sentences.shape[0]
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths).transpose(1, 0)  # shape: (b, len, K)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, max(sen_lengths)):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device


class NER:
    def __init__(self, weights_url, tag_vocab_url, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2", device=device)
        self.rubert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to(device)

        weights_file = wget.download(weights_url, out='model.pth')
        tag_vocab_file = wget.download(tag_vocab_url, out='tag_vocab.json')

        self.model = BiLSTMCRF.load(weights_file, device)
        self.tag_vocab = Vocab.load(tag_vocab_file)

    def get_ne(self, texts, ids):
        tokenized = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
        embded = self.rubert_model(**tokenized)['last_hidden_state'].to(self.device)
        mask = tokenized['attention_mask'].to(self.device)
        sent_lengths = mask.sum(1)
        tokens = self.model.predict(embded, mask, sent_lengths)
        return self.align(tokenized, tokens, texts, ids)

    def align(self, tokenized, tokens, texts, ids):
        out = []
        for i, id_ in enumerate(ids):
            words = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][i])
            tags = [self.tag_vocab.id2word(j) for j in tokens[i]]

            new_text = texts[i]
            cur = 0
            is_span = False
            category = ''
            beginning = None

            for word, tag in list(zip(words, tags))[1:]:
                new_word = word.lstrip('##')
                start = new_text.find(new_word)
                end = start + len(new_word)
                if is_span and beginning is not None and \
                            (tag.split('-')[0] != 'I' or tag.split('-')[-1] != category):
                    out.append((id_, texts[i][beginning:cur], beginning, cur,
                                category))
                    is_span = False

                if tag.split('-')[0] == 'B':
                    beginning = cur + start
                    category = tag.split('-')[1]
                    is_span = True

                cur += end
                new_text = new_text[end:]
        return out


if __name__ == '__main__':
    weights = 'https://github.com/PhilBurub/NLPcourse_HSE/raw/main/Project/model.pth'
    tags = 'https://raw.githubusercontent.com/PhilBurub/NLPcourse_HSE/main/Project/tag_vocab.json'
    ner_model = NER(weights, tags)

    text = ("Не рекомендуем сие заведение от слова совсем. Позвонили забронировать столик. Нам сказали -да на 23:00 "
            "вечера столик за Вами. Приезжаем в предвкушении повеселиться. Охранник не пускает у него нет информации "
            "что столик забронирован. Более того что у него даже не было попытки прояснить ситуацию элементарно "
            "вызвав администратора. Это всё свидетельствует о странности сотрудников сея заведения.")
    print(ner_model.get_ne([text], [3382]))
