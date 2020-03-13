import torch
import torch.utils.data
import os
import numpy as np
import configparser
import multiprocessing
import json
import pandas as pd
from subprocess import call
import sentencepiece as spm

class Config:
	def __init__(self):
		pass

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")

	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

	#[model]
	config.num_tokens=int(parser.get("model", "num_tokens"))
	config.num_decoder_layers=int(parser.get("model", "num_decoder_layers"))
	config.num_decoder_hidden=int(parser.get("model", "num_decoder_hidden"))
	config.tokenizer_training_text_path=parser.get("model", "tokenizer_training_text_path")

	#[training]
	config.base_path=parser.get("training", "base_path")
	config.lr=float(parser.get("training", "lr"))
	config.lr_period=int(parser.get("training", "lr_period"))
	config.gamma=float(parser.get("training", "gamma"))
	config.batch_size=int(parser.get("training", "batch_size"))
	config.num_epochs=int(parser.get("training", "num_epochs"))

	#[inference]
	config.beam_width=int(parser.get("inference", "beam_width"))

	return config

def get_ASR_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	base_path = config.base_path

	# Get dfs
	train_df = pd.read_csv(os.path.join(base_path, "train_data.csv"))
	valid_df = pd.read_csv(os.path.join(base_path, "valid_data.csv"))
	test_df = pd.read_csv(os.path.join(base_path, "test_data.csv"))

	# Create dataset objects
	train_dataset = TextDataset(train_df, config) #, tokenizer_sampling=True)
	valid_dataset = TextDataset(valid_df, config)
	test_dataset = TextDataset(test_df, config)

	return train_dataset, valid_dataset, test_dataset

class TextDataset(torch.utils.data.Dataset):
	def __init__(self, df, config, tokenizer_sampling=False):
		"""
		df: dataframe of texts
		config: Config object (contains info about model and training)
		"""
		# dataframe with wav file paths, transcripts
		self.df = df

		# get tokenizer
		num_tokens = config.num_tokens
		self.base_path = config.base_path
		tokenizer_model_prefix = "tokenizer_" + str(num_tokens) + "_tokens"
		tokenizer_path = os.path.join(self.base_path, tokenizer_model_prefix + ".model")
		tokenizer = spm.SentencePieceProcessor()

		# if tokenizer exists, load it
		try:
			print("Loading tokenizer from " + tokenizer_path)
			tokenizer.Load(tokenizer_path)

		# if not, create it
		except OSError:
			print("Tokenizer not found. Building tokenizer from training labels.")

			# create txt file needed by tokenizer training
			txt_path = os.path.join(self.base_path, config.tokenizer_training_text_path)

			# train tokenizer
			spm.SentencePieceTrainer.Train('--input=' + txt_path + ' --model_prefix=' + tokenizer_model_prefix + ' --vocab_size=' + str(num_tokens) + ' --hard_vocab_limit=false')

			# move tokenizer to base_path
			call("mv " + tokenizer_model_prefix + ".vocab " + self.base_path, shell=True)
			call("mv " + tokenizer_model_prefix + ".model " + self.base_path, shell=True)

			# load it
			tokenizer.Load(tokenizer_path)

		self.tokenizer = tokenizer
		self.tokenizer_sampling = tokenizer_sampling
		if self.tokenizer_sampling: print("Using tokenizer sampling")
		self.loader = torch.utils.data.DataLoader(self, batch_size=config.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=Collate())
		self.add_eos = True

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		if not self.tokenizer_sampling: y = self.tokenizer.EncodeAsIds(self.df.transcript[idx])
		if self.tokenizer_sampling: y = self.tokenizer.SampleEncodeAsIds(self.df.transcript[idx], -1, 0.1)
		if self.add_eos: y += self.tokenizer.eos_id()
		return (y, idx)

class Collate:
	def __init__(self):
		pass

	def __call__(self, batch):
		"""
		batch: list of tuples

		"""
		y = []; idxs = []
		batch_size = len(batch)
		for index in range(batch_size):
			y_,idx = batch[index]

		batch_size = len(idxs) # in case we threw some away

		# pad all sequences to have same length
		U = [len(y_) for y_ in y]
		U_max = max(U)
		for index in range(batch_size):
			y_pad_length = (U_max - len(y[index]))
			y[index] = torch.nn.functional.pad(y[index], (0,y_pad_length), value=0)

		y = torch.stack(y)

		return (y,U,idxs)
