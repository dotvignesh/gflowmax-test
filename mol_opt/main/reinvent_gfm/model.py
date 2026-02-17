#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Variable


class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.flow_head = nn.Linear(512, 1)
        self.linear = nn.Linear(512, voc_size)

    def step(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        logits = self.linear(x)
        log_flow = self.flow_head(x).squeeze(-1)
        return logits, h_out, log_flow

    def forward(self, x, h):
        logits, h_out, _ = self.step(x, h)
        return logits, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 512))


class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        print('need to wait several minutes')
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def _device_for(self, net=None):
        if net is None:
            net = self.rnn
        return next(net.parameters()).device

    def _prefix_batch_inputs(self, prefixes, net=None):
        if len(prefixes) == 0:
            return None, None
        device = self._device_for(net)
        go = self.voc.vocab['[GO]']
        eos = self.voc.vocab['[EOS]']
        lengths = torch.tensor([len(p) + 1 for p in prefixes], dtype=torch.long, device=device)
        max_len = int(lengths.max().item())
        inputs = torch.full((len(prefixes), max_len), eos, dtype=torch.long, device=device)
        inputs[:, 0] = go
        for i, prefix in enumerate(prefixes):
            if len(prefix) > 0:
                inputs[i, 1:1 + len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
        return inputs, lengths

    def _prefix_outputs(self, prefixes, net=None):
        if net is None:
            net = self.rnn
        if len(prefixes) == 0:
            device = self._device_for(net)
            return (
                torch.empty((0, self.voc.vocab_size), device=device),
                torch.empty((0,), device=device),
            )

        inputs, lengths = self._prefix_batch_inputs(prefixes, net=net)
        batch_size, max_len = inputs.size()
        h = net.init_h(batch_size)
        logits_steps = []
        flow_steps = []
        for step in range(max_len):
            logits_t, h, logf_t = net.step(inputs[:, step], h)
            logits_steps.append(logits_t)
            flow_steps.append(logf_t)

        logits_steps = torch.stack(logits_steps, dim=1)
        flow_steps = torch.stack(flow_steps, dim=1)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits_steps.size(-1))
        logits = logits_steps.gather(1, gather_idx).squeeze(1)
        logf = flow_steps.gather(1, (lengths - 1).view(-1, 1)).squeeze(1)
        return logits, logf

    def transition_logpf_logf(self, prefixes, actions, net=None):
        logits, logf = self._prefix_outputs(prefixes, net=net)
        if len(prefixes) == 0:
            return logf, logf
        device = logits.device
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        log_probs = F.log_softmax(logits, dim=1)
        logpf = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
        return logpf, logf

    def prefix_logf(self, prefixes, net=None):
        _, logf = self._prefix_outputs(prefixes, net=net)
        return logf

    def likelihood(self, target):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenght) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['[GO]']
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = Variable(torch.zeros(batch_size).float())
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy

    def sample(self, batch_size, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['[GO]']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits, dim = 1)
            log_prob = F.log_softmax(logits, dim = 1)
            x = torch.multinomial(prob, num_samples=1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['[EOS]']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

    def sample_with_trace(self, batch_size, max_length=140):
        """Sample a batch of sequences while preserving per-sequence termination."""
        start_token = Variable(torch.zeros(batch_size).long())
        start_token[:] = self.voc.vocab['[GO]']
        h = self.rnn.init_h(batch_size)
        x = start_token
        eos_token = self.voc.vocab['[EOS]']

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        entropy = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self._device_for())

        for _ in range(max_length):
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            x_next = torch.multinomial(prob, num_samples=1).view(-1)
            x_next = torch.where(finished, torch.full_like(x_next, eos_token), x_next)
            sequences.append(x_next.view(-1, 1))

            active = (~finished).float()
            log_probs += NLLLoss(log_prob, x_next) * active
            entropy += -torch.sum((log_prob * prob), 1) * active
            finished = finished | (x_next == eos_token)
            x = Variable(x_next.data)
            if bool(torch.all(finished)):
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy


def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
