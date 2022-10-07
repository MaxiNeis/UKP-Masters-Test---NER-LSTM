import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Tuple

from preprocessing import run_pp_pipeline


on_gpu = False
if torch.cuda.is_available():
  cuda = torch.device('cuda')
  on_gpu = True


word2idx_vocab, LABEL_TO_ID, embeddings_matrix, max_seq_length, train_dataloader, dev_dataloader, dev_label, test_dataloader, test_label = run_pp_pipeline(on_gpu)

lstm_hidden_size=100
number_of_tags=len(LABEL_TO_ID)
learning_rate=0.01


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        model.zero_grad()
        tag_scores = model(inputs)
        loss = loss_function(tag_scores.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  loss after {} training examples: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss


def create_emb_layer(embeddings_matrix, non_trainable=False):
    num_embeddings, embedding_dim = embeddings_matrix.shape
    if on_gpu:
      emb_layer = nn.Embedding.from_pretrained(torch.cuda.FloatTensor(embeddings_matrix))
    else:
      emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


def get_f1_prediction_tensor(dataset):
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataset:
            # dim: batch_size=1 x batch_max_len = [1, 59]
            outputs=model(inputs)
            outputs = outputs.squeeze()
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            predictions.append(preds)     
    return torch.LongTensor(predictions)

class LSTMTagger(nn.Module):

    def __init__(self, embeddings_matrix, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings_matrix = embeddings_matrix

        self.word_embeddings, num_embeddings, self.embedding_dim = create_emb_layer(embeddings_matrix, True)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        # dim: batch_size=1 x batch_max_len = [1, 59]
        embeds = self.word_embeddings(sentence)
        # dim: batch_size x batch_max_len x embedding_dim = [1, 59, 50]
        lstm_out, _ = self.lstm(embeds)
        # dim: batch_size x batch_max_len x embedding_dim = [1, 59, 50]
        tag_space = self.hidden2tag(lstm_out)
        # dim: batch_size x batch_max_len x #tags = [1, 59, 9]
        tag_scores = F.log_softmax(tag_space, dim=1)
        # dim: batch_size x batch_max_len x #tags = [1, 59, 9]
        #preds = tag_scores.squeeze()
        #_, preds = torch.max(preds, 1)
        return tag_scores


# Reference for F1 implementation: https://stackoverflow.com/a/63358412
class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """
    def __init__(self, average: str = 'weighted'):
        """
        Init.
        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label
        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)
        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.
        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
        Returns:
            f1 score
        """
        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)
        f1_score = 0
        for label_id in range(0, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)
            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1
        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))
        return f1_score


# Prepare parameter 
model = LSTMTagger(embeddings_matrix, lstm_hidden_size, len(word2idx_vocab), len(LABEL_TO_ID))
if on_gpu:
  model.to(cuda)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0
EPOCHS = 20
best_v_loss = 1_000_000.


for epoch in range(EPOCHS):
    print('\nEPOCH {}:'.format(epoch_number + 1))
    print(20*"=" + "\n")

    if epoch_number == 0:
        with torch.no_grad():
            inputs, labels = next(iter(train_dataloader))
            tag_scores = model(inputs)

    # Make sure training mode is on
    model.train(True)
    avg_loss = train_one_epoch()

    # Training mode not needed for reporting
    model.train(False)

    running_v_loss = 0.0
    for i, v_data in enumerate(dev_dataloader):
        v_inputs, v_labels = v_data
        v_tag_scores = model(v_inputs)
        v_loss = loss_function(v_tag_scores.squeeze(), v_labels.squeeze())
        running_v_loss += v_loss

    # Calculate and print F1_score(dev) after training for one epoch
    f1_metric = F1Score("macro")
    f1_dev = f1_metric(get_f1_prediction_tensor(dev_dataloader), dev_label.squeeze().cpu())

    avg_v_loss = running_v_loss / (i + 1)
    print('Loss train: {}   Loss val: {}'.format(avg_loss, avg_v_loss))
    print('F1 score on dev: ' + str(f1_dev.item()))

    # Track best performance, and save the model's state
    if avg_v_loss < best_v_loss:
        best_v_loss = avg_v_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

f1_metric_test = F1Score("macro")
f1_test = f1_metric_test(get_f1_prediction_tensor(test_dataloader), test_label.squeeze().cpu())
print('F1 score on test: ' + str(f1_test.item()))