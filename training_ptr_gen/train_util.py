import numpy as np
import torch
from data_util import config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input_from_batch(batch):
    """
    :param batch: a numpy version of encoder batch
    :return: a torch version of encoder batch including some initialization
    """
    batch_size = len(batch.enc_lens)
    enc_batch = torch.from_numpy(batch.enc_batch).to(device, dtype=torch.long)
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask).to(device)
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None
    coverage = None

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim)).to(device)

    if config.pointer_gen:
        enc_batch_extend_vocab = torch.from_numpy(batch.enc_batch_extend_vocab).to(device, dtype=torch.long)
        if batch.max_art_oov > 0:
            extra_zeros = torch.zeros(batch_size, batch.max_art_oov).to(device)

    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size()).to(device)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


def get_output_from_batch(batch):
    """
    :param batch: a numpy version of decoder batch
    :return: a torch version of decoder batch including some initialization
    """
    dec_batch = torch.from_numpy(batch.dec_batch).to(device, dtype=torch.long)
    dec_padding_mask = torch.from_numpy(batch.dec_padding_mask).to(device)
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = torch.from_numpy(dec_lens).to(device, dtype=torch.float)
    target_batch = torch.from_numpy(batch.target_batch).to(device, dtype=torch.long)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

