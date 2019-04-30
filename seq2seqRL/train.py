from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Required packages
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
# import torch.nn.functional as F
import csv
import random
import os
from io import open
import math
import coloredlogs, logging
# Local files
from model.utils import *
from model.voc import *
from model.model import *
from model.attention import *
from model.encoder import *
from model.decoder import *
from model.greedyDecoder import *
from model.hyperparameters import *




class Trainer:
	def __init__(self):
		self.corpus_name = "cornell movie-dialogs corpus"
		self.corpus = os.path.join("data", self.corpus_name)
		# Define path to new file
		self.datafile = os.path.join(self.corpus, "formatted_movie_lines.txt")
		self.trainer_logger = logging.getLogger('Trainer')
		coloredlogs.install(fmt='[%(asctime)s][%(name)s][%(levelname)s]: %(message)s',
			level='DEBUG',logger=self.trainer_logger)

		# Load/Assemble voc and pairs
		self.save_dir = os.path.join("data", "save")
		self.voc, self.pairs = loadPrepareData(self.corpus, self.corpus_name, self.datafile, self.save_dir, 
			logger=self.trainer_logger)
		self.trainer_logger.debug('Print some pairs to validate:')
		for pair in self.pairs[:10]:
		    print(pair)

	def init(self, loadFromFile=False):
		# Load lines and process conversations
		self.trainer_logger.info('Processing corpus...')
		lines = loadLines(os.path.join(self.corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
		self.trainer_logger.info('Loading conversations...')
		conversations = loadConversations(os.path.join(self.corpus, "movie_conversations.txt"),
		                                  lines, MOVIE_CONVERSATIONS_FIELDS)
		# Write new csv file
		self.trainer_logger.info('Writing newly formatted file...')
		with open(self.datafile, 'w', encoding='utf-8') as outputfile:
		    writer = csv.writer(outputfile, delimiter=delimiter)
		    for pair in extractSentencePairs(conversations):
		        writer.writerow(pair)

		

		# Trim voc and pairs
		self.pairs = trimRareWords(self.voc, self.pairs, MIN_COUNT, logger=self.trainer_logger)

		# Example for validation
		small_batch_size = 5
		batches = batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(small_batch_size)])
		input_variable, lengths, target_variable, mask, max_target_len = batches


		self.loadFilename = os.path.join(self.save_dir, model_name, self.corpus_name,
		                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
		                           '{}_checkpoint.tar'.format(checkpoint_iter))

		self.checkpoint = None
		# Load model if a loadFilename is provided
		if loadFromFile:
		    # If loading on same machine the model was trained on
		    self.checkpoint = torch.load(loadFilename)
		    # If loading a model trained on GPU to CPU
		    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
		    encoder_sd = checkpoint['en']
		    decoder_sd = checkpoint['de']
		    encoder_optimizer_sd = checkpoint['en_opt']
		    decoder_optimizer_sd = checkpoint['de_opt']
		    embedding_sd = checkpoint['embedding']
		    self.voc.__dict__ = checkpoint['voc_dict']


		self.trainer_logger.info('Building encoder and decoder ...')
		# Initialize word embeddings
		self.embedding = nn.Embedding(self.voc.num_words, hidden_size)
		if loadFromFile:
		    self.embedding.load_state_dict(embedding_sd)
		# Initialize encoder & decoder models
		self.encoder = EncoderRNN(hidden_size, self.embedding, encoder_n_layers, dropout)
		self.decoder = LuongAttnDecoderRNN(attn_model, self.embedding, hidden_size, 
			self.voc.num_words, decoder_n_layers, dropout)
		if loadFromFile:
		    self.encoder.load_state_dict(encoder_sd)
		    self.decoder.load_state_dict(decoder_sd)
		# Use appropriate device
		self.encoder = self.encoder.to(device)
		self.decoder = self.decoder.to(device)
		self.trainer_logger.info('Models built and ready to go!')



		# Ensure dropout layers are in train mode
		self.encoder.train()
		self.decoder.train()

		# Initialize optimizers
		self.trainer_logger.info('Building optimizers ...')
		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
		if loadFromFile:
		    self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
		    self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)

	def train(self, n_iteration=n_iteration):
		# Run training iterations
		self.trainer_logger.info("Starting Training!")
		trainIters(model_name, self.voc, self.pairs, self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer,
		           self.embedding, encoder_n_layers, decoder_n_layers, self.save_dir, n_iteration, batch_size,
		           print_every, save_every, clip, self.corpus_name, self.loadFilename, logger=self.trainer_logger)
		# Set dropout layers to eval mode
		self.encoder.eval()
		self.decoder.eval()


		# Initialize search module
		self.searcher = GreedySearchDecoder(self.encoder, self.decoder)

	def evaluating_samples(self):
		# Begin chatting (uncomment and run the following line to begin)
		evaluateInput(self.encoder, self.decoder, self.searcher, self.voc)

trainer = Trainer()
trainer.init()
trainer.train()
trainer.evaluating_samples()
