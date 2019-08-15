"""
The initialization only uses the premise and hypothesis embeddings but not the diff_product
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append("..")
from mutils import get_keys_from_vals, assert_sizes


def array_all_true(arr):
	for i in arr:
		if i == False:
			return False
	return True

"""
AttentionDecoder for the explanation
"""
class AttentionDecoder(nn.Module):
	def __init__(self, config):
		super(AttentionDecoder, self).__init__()
		
		self.decoder_type = config['decoder_type']
		self.word_emb_dim = config['word_emb_dim']
		self.dec_rnn_dim = config['dec_rnn_dim']
		self.enc_rnn_dim = config['enc_rnn_dim']
		self.dpout_dec = config['dpout_dec']
		self.n_vocab = config['n_vocab']
		self.word_index = config['word_index']
		self.word_vec = config['word_vec']
		self.max_T_decoder = config['max_T_decoder']
		self.max_T_encoder = config['max_T_encoder']
		self.n_layers_dec = config['n_layers_dec']
		# for decoder intial state
		self.use_init = config['use_init']
		# attention type: dot product or linear layer
		self.att_type = config['att_type'] # 'lin' or 'dot'
		# whether to visualize attention weights
		self.att_hid_dim =config['att_hid_dim']


		self.sent_dim = 2 * config['enc_rnn_dim']
		if config['encoder_type'] in ["ConvNetEncoder", "InnerAttentionMILAEncoder"]:
			self.sent_dim = 4 * self.sent_dim 
		if config['encoder_type'] == "LSTMEncoder":
			self.sent_dim = self.sent_dim / 2 

		assert self.sent_dim == 4096, str(self.sent_dim)
		# TODO: remove this when implemented linear attention
		assert self.att_type == 'dot'

		self.context_proj = nn.Linear(4 * self.sent_dim, self.dec_rnn_dim)

		self.att_ht_proj1 = nn.Sequential(
				nn.Linear(self.sent_dim, self.att_hid_dim),
				nn.Tanh(),
				)

		self.att_context_proj1 = nn.Sequential(
				nn.Linear(self.dec_rnn_dim, self.att_hid_dim),
				nn.Tanh(),
				)

		self.att_ht_before_weighting_proj1 = nn.Sequential(
				nn.Linear(self.sent_dim, self.att_hid_dim),
				nn.Tanh(),
				)

		self.att_ht_proj2 = nn.Sequential(
				nn.Linear(self.sent_dim, self.att_hid_dim),
				nn.Tanh(),
				)

		self.att_context_proj2 = nn.Sequential(
				nn.Linear(self.dec_rnn_dim, self.att_hid_dim),
				nn.Tanh(),
				)

		self.att_ht_before_weighting_proj2 = nn.Sequential(
				nn.Linear(self.sent_dim, self.att_hid_dim),
				nn.Tanh(),
				)


		self.proj_inp_dec = nn.Linear(2 * self.att_hid_dim + self.word_emb_dim, self.dec_rnn_dim)	
		if self.decoder_type == 'gru':
			self.decoder_rnn = nn.GRU(self.dec_rnn_dim, self.dec_rnn_dim, self.n_layers_dec, bidirectional=False, dropout=self.dpout_dec)
		else: # 'lstm'
			self.decoder_rnn = nn.LSTM(self.dec_rnn_dim, self.dec_rnn_dim, self.n_layers_dec, bidirectional=False, dropout=self.dpout_dec)

		# att softmax
		self.softmax_att = nn.Softmax(2)

		# vocab layer
		self.vocab_layer = nn.Linear(self.dec_rnn_dim, self.n_vocab)


	def forward(self, expl, enc_out_s1, enc_out_s2, s1_embed, s2_embed, mode, visualize):
		# expl: Variable(seqlen x bsize x worddim)
		# s1/2_embed: Variable(bsize x sent_dim)
		
		assert mode in ['forloop', 'teacher'], mode

		current_T_dec = expl.size(0)
		batch_size = expl.size(1)
		assert_sizes(s1_embed, 2, [batch_size, self.sent_dim])
		assert_sizes(s2_embed, 2, [batch_size, self.sent_dim])
		assert_sizes(expl, 3, [current_T_dec, batch_size, self.word_emb_dim])
		assert_sizes(enc_out_s1, 3, [self.max_T_encoder, batch_size, 2 * self.enc_rnn_dim])
		assert_sizes(enc_out_s2, 3, [self.max_T_encoder, batch_size, 2 * self.enc_rnn_dim])

		context = torch.cat([s1_embed, s2_embed, torch.abs(s1_embed - s2_embed), s1_embed * s2_embed], 1).unsqueeze(0)
		assert_sizes(context, 3, [1, batch_size, 4 * self.sent_dim])

		# init decoder
		if self.use_init:
			init_0 = self.context_proj(context).expand(self.n_layers_dec, batch_size, self.dec_rnn_dim)
		else:
			init_0 = Variable(torch.zeros(self.n_layers_dec, batch_size, self.dec_rnn_dim)).cuda()

		init_state = init_0
		if self.decoder_type == 'lstm':
			init_state = (init_0, init_0)

		self.decoder_rnn.flatten_parameters()

		out_expl = None
		state_t = init_state
		context = self.context_proj(context)
		if mode == "teacher":
			for t_dec in range(current_T_dec):
				# attention over premise
				context1 = self.att_context_proj1(context).permute(1, 0, 2)
				assert_sizes(context1, 3, [batch_size, 1, self.att_hid_dim])
				
				inp_att_1 = self.att_ht_proj1(enc_out_s1).transpose(1,0).transpose(2,1)
				assert_sizes(inp_att_1, 3, [batch_size, self.att_hid_dim, self.max_T_encoder])
				
				dot_prod_att_1 = torch.bmm(context1, inp_att_1)
				assert_sizes(dot_prod_att_1, 3, [batch_size, 1, self.max_T_encoder])
				
				att_weights_1 = self.softmax_att(dot_prod_att_1)
				assert_sizes(att_weights_1, 3, [batch_size, 1, self.max_T_encoder])
				
				att_applied_1 = torch.bmm(att_weights_1, self.att_ht_before_weighting_proj1(enc_out_s1).permute(1, 0, 2))
				assert_sizes(att_applied_1, 3, [batch_size, 1, self.att_hid_dim])

				att_applied_perm_1 = att_applied_1.permute(1, 0, 2)
				assert_sizes(att_applied_perm_1, 3, [1, batch_size, self.att_hid_dim])

				# attention over hypothesis
				context2 = self.att_context_proj2(context).permute(1, 0, 2)
				assert_sizes(context2, 3, [batch_size, 1, self.att_hid_dim])

				inp_att_2 = self.att_ht_proj2(enc_out_s2).transpose(1,0).transpose(2,1)
				assert_sizes(inp_att_2, 3, [batch_size, self.att_hid_dim, self.max_T_encoder])
				
				dot_prod_att_2 = torch.bmm(context2, inp_att_2)
				assert_sizes(dot_prod_att_2, 3, [batch_size, 1, self.max_T_encoder])
				
				att_weights_2 = self.softmax_att(dot_prod_att_2)
				assert_sizes(att_weights_2, 3, [batch_size, 1, self.max_T_encoder])
				
				att_applied_2 = torch.bmm(att_weights_2, self.att_ht_before_weighting_proj2(enc_out_s2).permute(1, 0, 2))
				assert_sizes(att_applied_2, 3, [batch_size, 1, self.att_hid_dim])

				att_applied_perm_2 = att_applied_2.permute(1, 0, 2)
				assert_sizes(att_applied_perm_2, 3, [1, batch_size, self.att_hid_dim])
				
				input_dec = torch.cat([expl[t_dec].unsqueeze(0), att_applied_perm_1, att_applied_perm_2], 2) 
				input_dec = nn.Dropout(self.dpout_dec)(self.proj_inp_dec(input_dec))

				out_dec, state_t = self.decoder_rnn(input_dec, state_t)
				assert_sizes(out_dec, 3, [1, batch_size, self.dec_rnn_dim])
				if self.decoder_type == 'lstm':
					context = state_t[0]
				else:
					context = state_t

				if out_expl is None:
					out_expl = out_dec
				else:
					out_expl = torch.cat([out_expl, out_dec], 0)

			out_expl = self.vocab_layer(out_expl)
			assert_sizes(out_expl, 3, [current_T_dec, batch_size, self.n_vocab])
			return out_expl

		else:
			pred_expls = []
			finished = []
			for i in range(batch_size):
				pred_expls.append("")
				finished.append(False)

			t_dec = 0
			word_t = expl[0].unsqueeze(0)
			while t_dec < self.max_T_decoder and not array_all_true(finished):
				#print "\n\n\n t: ", t_dec

				assert_sizes(word_t, 3, [1, batch_size, self.word_emb_dim])
				word_embed = torch.zeros(1, batch_size, self.word_emb_dim)
				
				# attention over premise
				context1 = self.att_context_proj1(context).permute(1, 0, 2)
				assert_sizes(context1, 3, [batch_size, 1, self.att_hid_dim])
				
				inp_att_1 = self.att_ht_proj1(enc_out_s1).transpose(1,0).transpose(2,1)
				assert_sizes(inp_att_1, 3, [batch_size, self.att_hid_dim, self.max_T_encoder])
				
				dot_prod_att_1 = torch.bmm(context1, inp_att_1)
				assert_sizes(dot_prod_att_1, 3, [batch_size, 1, self.max_T_encoder])
				
				att_weights_1 = self.softmax_att(dot_prod_att_1)
				assert_sizes(att_weights_1, 3, [batch_size, 1, self.max_T_encoder])
				
				att_applied_1 = torch.bmm(att_weights_1, self.att_ht_before_weighting_proj1(enc_out_s1).permute(1, 0, 2))
				assert_sizes(att_applied_1, 3, [batch_size, 1, self.att_hid_dim])

				att_applied_perm_1 = att_applied_1.permute(1, 0, 2)
				assert_sizes(att_applied_perm_1, 3, [1, batch_size, self.att_hid_dim])

				# attention over hypothesis
				context2 = self.att_context_proj2(context).permute(1, 0, 2)
				assert_sizes(context2, 3, [batch_size, 1, self.att_hid_dim])

				inp_att_2 = self.att_ht_proj2(enc_out_s2).transpose(1,0).transpose(2,1)
				assert_sizes(inp_att_2, 3, [batch_size, self.att_hid_dim, self.max_T_encoder])
				
				dot_prod_att_2 = torch.bmm(context2, inp_att_2)
				assert_sizes(dot_prod_att_2, 3, [batch_size, 1, self.max_T_encoder])
				
				att_weights_2 = self.softmax_att(dot_prod_att_2)
				assert_sizes(att_weights_2, 3, [batch_size, 1, self.max_T_encoder])
				
				att_applied_2 = torch.bmm(att_weights_2, self.att_ht_before_weighting_proj2(enc_out_s2).permute(1, 0, 2))
				assert_sizes(att_applied_2, 3, [batch_size, 1, self.att_hid_dim])

				att_applied_perm_2 = att_applied_2.permute(1, 0, 2)
				assert_sizes(att_applied_perm_2, 3, [1, batch_size, self.att_hid_dim])
				
				input_dec = torch.cat([word_t, att_applied_perm_1, att_applied_perm_2], 2) 
				input_dec = self.proj_inp_dec(input_dec)
				
				#print "att_weights_1[0] ", att_weights_1[0]
				#print "att_weights_2[0] ", att_weights_2[0]

				# get one visualization from the current batch
				if visualize:
					if t_dec == 0:
						weights_1 = att_weights_1[0]
						weights_2 = att_weights_2[0]
					else:
						weights_1 = torch.cat([weights_1, att_weights_1[0]], 0)
						weights_2 = torch.cat([weights_2, att_weights_2[0]], 0)

				for ii in range(batch_size):
					assert abs(att_weights_1[ii].data.sum() - 1) < 1e-5, str(att_weights_1[ii].data.sum())
					assert abs(att_weights_2[ii].data.sum() - 1) < 1e-5, str(att_weights_2[ii].data.sum())

				out_t, state_t = self.decoder_rnn(input_dec, state_t)
				assert_sizes(out_t, 3, [1, batch_size, self.dec_rnn_dim])
				out_t = self.vocab_layer(out_t)
				if self.decoder_type == 'lstm':
					context = state_t[0]
				else:
					context = state_t

				i_t = torch.max(out_t, 2)[1].data
				assert_sizes(i_t, 2, [1, batch_size])
				pred_words = get_keys_from_vals(i_t, self.word_index) # array of bs of words at current timestep
				assert len(pred_words) == batch_size, "pred_words " + str(len(pred_words)) + " batch_size " + str(batch_size)
				for i in range(batch_size):
					if pred_words[i] == '</s>':
						finished[i] = True
					if not finished[i]:
						pred_expls[i] += " " + pred_words[i]
					word_embed[0, i] = torch.from_numpy(self.word_vec[pred_words[i]])
				word_t = Variable(word_embed.cuda())

				t_dec += 1
			
			if visualize:
				assert weights_1.dim() == 2
				assert weights_1.size(1) == self.max_T_encoder
				assert weights_2.dim() == 2
				assert weights_2.size(1) == self.max_T_encoder
				pred_expls = [pred_expls, weights_1, weights_2]
			return pred_expls



"""
BLSTM (max/mean) encoder
"""
class BLSTMEncoder(nn.Module):

	def __init__(self, config):
		super(BLSTMEncoder, self).__init__()
		self.bsize = config['bsize']
		self.word_emb_dim = config['word_emb_dim']
		self.enc_rnn_dim = config['enc_rnn_dim']
		self.pool_type = config['pool_type']
		self.dpout_enc = config['dpout_enc']
		self.max_T_encoder = config['max_T_encoder']

		self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_rnn_dim, 1,
								bidirectional=True, dropout=self.dpout_enc)

	def is_cuda(self):
		# either all weights are on cpu or they are on gpu
		return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

	def forward(self, sent_tuple):
		# sent_len: [max_len, ..., min_len] (bsize)
		# sent: Variable(seqlen x bsize x worddim)
		sent, sent_len = sent_tuple
		#assert_sizes(sent, 3, [self.max_T_encoder, sent.size(1), self.word_emb_dim])

		# Sort by length (keep idx)
		sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
			else torch.from_numpy(idx_sort)
		sent = sent.index_select(1, Variable(idx_sort))

		# Handling padding in Recurrent Networks
		sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
		self.enc_lstm.flatten_parameters()
		sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
		padding_value = 0.0
		if self.pool_type == "max":
			padding_value = -100
		sent_output_padding = nn.utils.rnn.pad_packed_sequence(sent_output, False, padding_value)[0]
		sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, False, 0)[0]

		# Un-sort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
			else torch.from_numpy(idx_unsort)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))
		sent_output_padding = sent_output_padding.index_select(1, Variable(idx_unsort))
		
		# Pooling
		if self.pool_type == "mean":
			sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
			emb = torch.sum(sent_output_padding, 0).squeeze(0)
			emb = emb / sent_len.expand_as(emb)
		elif self.pool_type == "max":
			emb = torch.max(sent_output_padding, 0)[0]
			if emb.ndimension() == 3:
				emb = emb.squeeze(0)
				assert emb.ndimension() == 2, "emb.ndimension()=" + str(emb.ndimension())

		# pad with zeros so that max length is the same for all, needed for attention
		if sent_output.size(0) < self.max_T_encoder:
			pad_tensor = Variable(torch.zeros(self.max_T_encoder - sent_output.size(0), sent_output.size(1), sent_output.size(2)).cuda())
			sent_output = torch.cat([sent_output, pad_tensor], 0)
		
		return sent_output, emb



	def set_glove_path(self, glove_path):
		self.glove_path = glove_path

	def get_word_dict(self, sentences, tokenize=True):
		# create vocab of words
		word_dict = {}
		if tokenize:
			from nltk.tokenize import word_tokenize
		sentences = [s.split() if not tokenize else word_tokenize(s)
					 for s in sentences]
		for sent in sentences:
			for word in sent:
				if word not in word_dict:
					word_dict[word] = ''
		word_dict['<s>'] = ''
		word_dict['</s>'] = ''
		return word_dict

	def get_glove(self, word_dict):
		assert hasattr(self, 'glove_path'), \
			   'warning : you need to set_glove_path(glove_path)'
		# create word_vec with glove vectors
		word_vec = {}
		with open(self.glove_path) as f:
			for line in f:
				word, vec = line.split(' ', 1)
				if word in word_dict:
					word_vec[word] = np.fromstring(vec, sep=' ')
		print('Found {0}(/{1}) words with glove vectors'.format(
					len(word_vec), len(word_dict)))
		return word_vec

	def get_glove_k(self, K):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		# create word_vec with k first glove vectors
		k = 0
		word_vec = {}
		with open(self.glove_path) as f:
			for line in f:
				word, vec = line.split(' ', 1)
				if k <= K:
					word_vec[word] = np.fromstring(vec, sep=' ')
					k += 1
				if k > K:
					if word in ['<s>', '</s>']:
						word_vec[word] = np.fromstring(vec, sep=' ')

				if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
					break
		return word_vec

	def build_vocab(self, sentences, tokenize=True):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		word_dict = self.get_word_dict(sentences, tokenize)
		self.word_vec = self.get_glove(word_dict)
		print('Vocab size from within BLSTMEncoder : {0}'.format(len(self.word_vec)))

	# build GloVe vocab with k most frequent words
	def build_vocab_k_words(self, K):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		self.word_vec = self.get_glove_k(K)
		print('Vocab size : {0}'.format(K))

	def update_vocab(self, sentences, tokenize=True):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
		word_dict = self.get_word_dict(sentences, tokenize)

		# keep only new words
		for word in self.word_vec:
			if word in word_dict:
				del word_dict[word]

		# udpate vocabulary
		if word_dict:
			new_word_vec = self.get_glove(word_dict)
			self.word_vec.update(new_word_vec)
		print('New vocab size : {0} (added {1} words)'.format(
						len(self.word_vec), len(new_word_vec)))

	def get_batch(self, batch):
		# sent in batch in decreasing order of lengths
		# batch: (bsize, max_len, word_dim)
		embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

		for i in range(len(batch)):
			for j in range(len(batch[i])):
				embed[j, i, :] = self.word_vec[batch[i][j]]

		return torch.FloatTensor(embed)

	def prepare_samples(self, sentences, bsize, tokenize, verbose):
		if tokenize:
			from nltk.tokenize import word_tokenize
		sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
					 ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
		n_w = np.sum([len(x) for x in sentences])

		# filters words without glove vectors
		for i in range(len(sentences)):
			s_f = [word for word in sentences[i] if word in self.word_vec]
			if not s_f:
				import warnings
				warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
							   Replacing by "</s>"..'.format(sentences[i], i))
				s_f = ['</s>']
			sentences[i] = s_f

		lengths = np.array([len(s) for s in sentences])
		n_wk = np.sum(lengths)
		if verbose:
			print('Nb words kept : {0}/{1} ({2} %)'.format(
						n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

		# sort by decreasing length
		lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
		sentences = np.array(sentences)[idx_sort]

		return sentences, lengths, idx_sort

	def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
		tic = time.time()
		sentences, lengths, idx_sort = self.prepare_samples(
						sentences, bsize, tokenize, verbose)

		embeddings = []
		for stidx in range(0, len(sentences), bsize):
			batch = Variable(self.get_batch(
						sentences[stidx:stidx + bsize]), volatile=True)
			if self.is_cuda():
				batch = batch.cuda()
			batch = self.forward(
				(batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
			embeddings.append(batch)
		embeddings = np.vstack(embeddings)

		# unsort
		idx_unsort = np.argsort(idx_sort)
		embeddings = embeddings[idx_unsort]

		if verbose:
			print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
					round(len(embeddings)/(time.time()-tic), 2),
					'gpu' if self.is_cuda() else 'cpu', bsize))
		return embeddings

	def visualize(self, sent, tokenize=True):
		if tokenize:
			from nltk.tokenize import word_tokenize

		sent = sent.split() if not tokenize else word_tokenize(sent)
		sent = [['<s>'] + [word for word in sent if word in self.word_vec] +
				['</s>']]

		if ' '.join(sent[0]) == '<s> </s>':
			import warnings
			warnings.warn('No words in "{0}" have glove vectors. Replacing \
						   by "<s> </s>"..'.format(sent))
		batch = Variable(self.get_batch(sent), volatile=True)

		if self.is_cuda():
			batch = batch.cuda()
		output = self.enc_lstm(batch)[0]
		output, idxs = torch.max(output, 0)
		# output, idxs = output.squeeze(), idxs.squeeze()
		idxs = idxs.data.cpu().numpy()
		argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

		# visualize model
		import matplotlib.pyplot as plt
		x = range(len(sent[0]))
		y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
		plt.xticks(x, sent[0], rotation=45)
		plt.bar(x, y)
		plt.ylabel('%')
		plt.title('Visualisation of words importance')
		plt.show()

		return output, idxs


"""
Main module for Natural Language Inference
"""
class eSNLIAttention(nn.Module):
	def __init__(self, config):
		super(eSNLIAttention, self).__init__()
		self.encoder_type = config['encoder_type']

		self.encoder = eval(self.encoder_type)(config)
		self.decoder = AttentionDecoder(config)

	def forward(self, s1, s2, expl, mode, visualize):
		# s1 : (s1, s1_len)
		# s2 : (s2, s2_len)
		# expl : Variable(T x bs x 300)

		u, u_emb = self.encoder(s1) # u = max_T_enc x bs x (2 * enc_dim) ; u_emb = 1 x bs x (2 * enc_dim)
		v, v_emb = self.encoder(s2) 

		out_expl = self.decoder(expl, u, v, u_emb, v_emb, mode, visualize)
		
		return out_expl

	def encode(self, s1):
		emb = self.encoder(s1)
		return emb