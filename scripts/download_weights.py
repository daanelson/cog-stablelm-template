#!/usr/bin/env python

import sys
sys.path.append('/src/')

from config import load_tokenizer, load_model

# pulls tokenizer
load_tokenizer()

# assumption - we're only running this to pull the model from HF for development setup
load_model()

