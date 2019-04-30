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
import pickle

# Local files
from json_encoder import *
from utils import *
from voc import *
from model import *
from hyperparameters import *

save_dir = os.path.join("data", "save")
corpus_name = "cornell cmovie-dialogs corpus"

# If you're loading your own model
# Set checkpoint to load from
checkpoint_iter = 4000


# Uncomment these lines if changes are made to the VOC data files
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))
# checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
voc = Voc(corpus_name)
# voc.__dict__ = checkpoint['voc_dict']
# with open('voc.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(voc.__dict__, f)

with open('voc.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    voc.__dict__ = pickle.load(f)

chat = torch.jit.load("scripted_chatbot.pth")


# For testing
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, None, None, chat, voc)
evaluateInput(None, None, chat, voc)
