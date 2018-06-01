import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np

torch.manual_seed(1)

# Input dim is 3, output dim is 3
lstm = nn.LSTM(3, 3)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# get data ready
def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, len(to_ix) - 2) for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerially stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# Use generator to load big file:
def read_in_block(file_path, BLOCK_SIZE=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            block = f.readlines(BLOCK_SIZE)
            if block:
                yield block
            else:
                return

# Load pretrained word_vector
def load_word_vector(vec_path):
    weight = []
    char2index_map = {}
    i = 0
    for block in read_in_block(vec_path):
        content = block[0].split(' ')
        if len(content) == 2:
            continue
        char2index_map.update({content[0]: i})
        weight.append(content[1:])
        i += 1
    embeddings_size = len(weight[0])
    rng = np.random.RandomState(23455)
    unknow = np.asarray(rng.normal(size=(embeddings_size)))
    padding = np.asarray(rng.normal(size=(embeddings_size)))
    weight.append(unknow)
    weight.append(padding)
    char2index_map.update({'unknow': i})
    char2index_map.update({'padding': i+1})
    weight = np.array(weight, dtype=float)
    return char2index_map, weight


class MyDataSet(Dataset):
    def __init__(self, source_path, target_path=None, transform=None):
        target_path = target_path
        sentences = []
        targets = []
        data_set = []
        for block in read_in_block(source_path, 1):
            content = block[0].strip().split(" ")
            sentences.append(content)
        if target_path:
            for block in read_in_block(target_path, 1):
                content = block[0].strip().split(" ")
                targets.append(content)
            assert len(sentences) == len(targets)
            for i in range(len(sentences)):
                data_set.append((sentences[i], targets[i]))
        else:
            for i in range(len(sentences)):
                data_set.append((sentences[i], []))

        self.data_set = data_set
        self.transform = transform

    def __getitem__(self, index):
        sentence, target = self.data_set[index]
        return sentence, target

    def __len__(self):
        return len(self.data_set)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, pretrained_word_vec):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        # self.tagset_size = len(tag_to_ix)
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pretrained_word_vec))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters. Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # self.transitions = nn.Parameter(torch.FloatTensor(pretrained_word_vec))

        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer from stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def _froward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = [] # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # The ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge(i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log_space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them ( we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # Now add in the emssion scores, and assign forward_var to the set of viterbi variables we just computed.
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag ( we dont want to return that to the caller)
        start = best_path.pop()
        assert  start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._froward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # Get the emission scores from BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def train(source_path, target_path):
    data_tf = transforms.Compose([transforms.ToTensor()])
    training_data = MyDataSet(source_path, target_path, transform=data_tf)
    tag_to_ix = {}

    target_path = "E:\hao.shen\Projects\\NER-master\\resource\\target_vocab.txt"
    with open(target_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            tag_to_ix.update({line.strip(): i})
            i += 1
        tag_to_ix.update({START_TAG: i})
        tag_to_ix.update({STOP_TAG: i+1})

    model = BiLSTM_CRF(len(CHAR2INDEX_MAP), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, PRETRAINED_WORD_VEC)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], CHAR2INDEX_MAP)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    for epoch in range(1):
        print("epoch" + str(epoch))
        for sentence, target in training_data:
            sentence_in = prepare_sequence(sentence, CHAR2INDEX_MAP)
            target = torch.tensor([tag_to_ix[t] for t in target], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, target)
            loss.backward()
            optimizer.step()
        print(loss)
    torch.save(model.state_dict(), "params.pk1")

def predict(predict_path):
    data_tf = transforms.Compose([transforms.ToTensor()])
    predict_data = MyDataSet(predict_path, transform=data_tf)
    tag_to_ix = {}
    target_path = "E:\hao.shen\Projects\\NER-master\\resource\\target_vocab.txt"
    with open(target_path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            tag_to_ix.update({line.strip(): i})
            i += 1
        tag_to_ix.update({START_TAG: i})
        tag_to_ix.update({STOP_TAG: i+1})
    model = BiLSTM_CRF(len(CHAR2INDEX_MAP), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, PRETRAINED_WORD_VEC)
    model.load_state_dict(torch.load("params.pk1"))
    with torch.no_grad():
        precheck_sent = prepare_sequence(predict_data[0][0], CHAR2INDEX_MAP)
        out = model(precheck_sent)
        new_tag_to_ix = {v: k for k, v in tag_to_ix.items()}
        for i in range(len(out)):
            print(precheck_sent[i], new_tag_to_ix[tag_to_ix[i]])




if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 4
    PRETRAINED_WORD_VEC_PATH = "E:\hao.shen\Projects\\NER-master\\resource\corpus.vec"
    CHAR2INDEX_MAP, PRETRAINED_WORD_VEC = load_word_vector(PRETRAINED_WORD_VEC_PATH)
    SOURCE_PATH = "E:\hao.shen\Projects\\NER-master\\resource\source.txt"
    TARGET_PATH = "E:\hao.shen\Projects\\NER-master\\resource\\target.txt"
    PREDICT_PATH = "E:\hao.shen\Projects\\NER-master\\resource\predict.txt"
    # train(SOURCE_PATH, target_path=TARGET_PATH)
    predict(PREDICT_PATH)

    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]
    #
    # word_to_ix = {}
    # for sentence, tags in training_data:
    #     for word in sentence:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    #
    # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    # model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, PRETRAINED_WORD_VEC)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    #
    # # Check predictions before training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #     print(model(precheck_sent))
    #
    # for epoch in range(300):
    #     for sentence, tags in training_data:
    #         model.zero_grad()
    #         sentence_in = prepare_sequence(sentence, word_to_ix)
    #         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
    #         loss = model.neg_log_likelihood(sentence_in, targets)
    #         loss.backward()
    #         optimizer.step()
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence("georgia reported that apple corporation made money".split(), word_to_ix)
    #     print(model(precheck_sent))

