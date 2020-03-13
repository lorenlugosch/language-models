import numpy as np
import torch
from tqdm import tqdm # for displaying progress bar
import os
import pandas as pd

class Trainer:
	def __init__(self, model, config):
		self.model = model
		self.config = config
		self.lr = config.lr
		self.lr_period = config.lr_period
		self.gamma = config.gamma
		self.checkpoint_path = os.path.join(self.config.folder, "training")
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_period, gamma=self.gamma)
		self.epoch = 0
		self.df = None
		if torch.cuda.is_available(): self.model.cuda()
		self.best_loss = np.inf

	def load_checkpoint(self):
		if os.path.isfile(os.path.join(self.checkpoint_path, "model_state.pth")):
			try:
				device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
				self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "model_state.pth"), map_location=device))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def load_best_model(self):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "best_model.pth"), map_location=device))

	def save_checkpoint(self, loss):
		try:
			torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "model_state.pth"))
			if loss < self.best_loss:
				self.best_loss = loss
				torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best_model.pth"))
		except:
			print("Could not save model")

	def log(self, results):
		if self.df is None:
			self.df = pd.DataFrame(columns=[field for field in results])
		self.df.loc[len(self.df)] = results
		self.df.to_csv(os.path.join(self.checkpoint_path, "log.csv"))

	def train(self, dataset, print_interval=100):
		train_loss = 0
		num_examples = 0
		self.model.train()
		for g in self.optimizer.param_groups:
			print("Current learning rate:", g['lr'])
		#self.model.print_frozen()
		for idx, batch in enumerate(tqdm(dataset.loader)):
			y,U,idxs = batch
			batch_size = len(y)
			log_probs = self.model(y,U)
			loss = -log_probs.mean()
			if torch.isnan(loss):
				print("nan detected!")
				sys.exit()
			self.optimizer.zero_grad()
			loss.backward()
			#clip_value = 5; torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
			self.optimizer.step()
			train_loss += loss.item() * batch_size
			num_examples += batch_size
			if idx % print_interval == 0:
				print("loss: " + str(loss.cpu().data.numpy().item()))
				y_sampled = self.model.sample(U[0])
				print("sample:", dataset.tokenizer.DecodeIds(y_sampled))
				print("")

		train_loss /= num_examples
		#self.model.unfreeze_one_layer()
		results = {"loss" : train_loss, "set": "train"}
		self.log(results)
		self.epoch += 1
		return train_loss

	def test(self, dataset, set):
		test_loss = 0
		num_examples = 0
		self.model.eval()
		for idx, batch in enumerate(tqdm(dataset.loader)): #enumerate(dataset.loader):
			y,U,_ = batch
			batch_size = len(x)
			num_examples += batch_size
			log_probs = self.model(x,y,T,U)
			loss = -log_probs.mean()
			test_loss += loss.item() * batch_size

		test_loss /= num_examples
		self.scheduler.step()
		results = {"loss" : test_loss, "set": set}
		self.log(results)
		return test_loss
