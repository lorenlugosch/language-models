import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import time

class DecoderRNN(torch.nn.Module):
	def __init__(self, num_decoder_layers, num_decoder_hidden, input_size, dropout):
		super(DecoderRNN, self).__init__()

		self.layers = []
		self.num_decoder_layers = num_decoder_layers
		for index in range(num_decoder_layers):
			if index == 0: 
				layer = torch.nn.GRUCell(input_size=input_size, hidden_size=num_decoder_hidden) 
			else:
				layer = torch.nn.GRUCell(input_size=num_decoder_hidden, hidden_size=num_decoder_hidden) 
			layer.name = "gru%d"%index
			self.layers.append(layer)

			layer = torch.nn.Dropout(p=dropout)
			layer.name = "dropout%d"%index
			self.layers.append(layer)
		self.layers = torch.nn.ModuleList(self.layers)

	def forward(self, input, previous_state):
		"""
		input: Tensor of shape (batch size, input_size)
		previous_state: Tensor of shape (batch size, num_decoder_layers, num_decoder_hidden)
		Given the input vector, update the hidden state of each decoder layer.
		"""
		# return self.gru(input, previous_state)

		state = []
		batch_size = input.shape[0]
		gru_index = 0
		for index, layer in enumerate(self.layers):
			if "gru" in layer.name:
				if index == 0:
					gru_input = input
				else:
					gru_input = layer_out
				layer_out = layer(gru_input, previous_state[:, gru_index])
				state.append(layer_out)
				gru_index += 1
			else:
				layer_out = layer(layer_out)
		state = torch.stack(state, dim=1)
		return state

class LanguageModel(torch.nn.Module):
	def __init__(self, config):
		super(LanguageModel, self).__init__()
		self.layers = []
		num_decoder_hidden = config.num_decoder_hidden
		num_decoder_layers = config.num_decoder_layers
		input_size = num_decoder_hidden

		self.initial_state = torch.nn.Parameter(torch.randn(num_decoder_layers,num_decoder_hidden))
		self.embed = torch.nn.Embedding(num_embeddings=config.num_tokens + 1, embedding_dim=num_decoder_hidden)
		self.rnn = DecoderRNN(num_decoder_layers, num_decoder_hidden, input_size, dropout=0.5)
		self.num_outputs = config.num_tokens + 1 # for blank symbol
		self.linear = torch.nn.Linear(num_decoder_hidden,self.num_outputs)
		self.start_symbol = self.num_outputs - 1 # blank index == start symbol
		self.use_label_smoothing = config.use_label_smoothing

	def forward_one_step(self, input, previous_state):
		embedding = self.embed(input)
		state = self.rnn.forward(embedding, previous_state)
		out = self.linear(state[:,-1])
		return out, state

	def forward(self, y, U):
		if next(self.parameters()).is_cuda:
			y = y.cuda()

		batch_size = y.shape[0]
		U_max = y.shape[1]
		outs = []
		state = torch.stack([self.initial_state] * batch_size).to(y.device)
		for u in range(U_max):
			if u == 0:
				decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
			else:
				decoder_input = y[:,u-1]
			out, state = self.forward_one_step(decoder_input, state)
			outs.append(out)
		out = torch.stack(outs, dim=1)

		log_probs = []
		for i in range(batch_size):
			log_prob = 
		return log_probs
