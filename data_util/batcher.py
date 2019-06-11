import queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

import random

import data_util.data as data
import data_util.config as config


random.seed(1234)


class Example(object):
    """
    example:

        article = "by . sam adams . published : . 04:35 est , 28 march 2013 . | . updated : . 08:04 est ," \
                  " 28 march 2013 . an ` aladdin \'s cave \' of stolen goods was unveiled today by police after" \
                  " they smashed a black market car parts racket operating on ebay . hundreds of parts were " \
                  "stripped from stolen vehicles then itemised and stored in a warehouse before being posted " \
                  "for sale on the internet auction site as ` scrapyard seconds . \' police raided ramzan " \
                  "ahmed \'s premises in bolton after noticing components sold by his online firm permanent " \
                  "discounts were suspiciously clean and undamaged - and discovered the stolen car parts " \
                  "together with a cannabis factory . the 34-year-old was sentenced to four years in prison " \
                  "after admitting cannabis production and 14 counts of handling stolen goods . theft : this " \
                  "warehouse full of stolen car parts in bolton was discovered after police smashed an illegal " \
                  "operation run by ramzan ahmed . operation : hundreds of parts were stripped from stolen vehicles " \
                  "then itemised and stored in the lock-up before being posted for sale on the internet auction " \
                  "site as ` scrapyard seconds \' find : among the parts found were wheels , engines , car " \
                  "seats , exhaust pipes , windscreens and chassis of more than a dozen stolen vehicles . police " \
                  "raided the former it worker \'s premises in bolton , greater manchester on march 17 , 2011 and " \
                  "uncovered a huge array of items looted from cars and other vehicles stolen across the north " \
                  "of england . they including wheels , engines , car seats , exhaust pipes , windscreens and " \
                  "chassis of more than a dozen stolen vehicles , which were in various stages of being taken " \
                  "apart . police searched ahmed , 34 , to find he had with him a set of keys to another unit " \
                  "next door . guilty : ramzan ahmed was jailed for four years after admitting to cannabis " \
                  "production and 14 counts of handling stolen goods . when they opened the second unit they" \
                  " discovered a large cannabis factory of 323 plants capable of producing a yield of almost " \
                  "13kg of cannabis . sold on the streets the cannabis was worth up to # 135,000 . ahmed \'s " \
                  "fingerprints were also found on cannabis production equipment inside the factory . details " \
                  "of the find emerged at bolton crown court where ahmed of bury was jailed for four years . " \
                  "vanessa thomson , prosecuting , told how ahmed first came to the attention of the police " \
                  "special investigation branch in 2011 after officers noticed components from relatively new " \
                  "cars were being sold on ebay by his company called permanent discounts . amongst the large" \
                  " number of car parts stored at unit 10 , were doors , tailgates , bumpers and other parts " \
                  "from at least 14 stolen vehicles , including a lexus , mazda rx8 , a toyota rav4 and three " \
                  "bmw 5 series . the total value of the stolen cars identified was at more than # 130,000 . " \
                  "shortly after the raids ahmed arrived only to drive away when he spotted the police . he was " \
                  "arrested a short time later on the m602 . in mitigation defence lawyer shirlie duckworth said " \
                  "the father-of-three ahmed had initially set up a legitimate car parts business after losing " \
                  "his job as an it worker with fujitsu . when it proved to be unsuccessful he found he was ` " \
                  "unable to resist \' when thieves approached him offering to sell him stolen cars . value : " \
                  "when they opened a second unit they discovered a large cannabis factory of 323 plants capable " \
                  "of producing a yield of almost 13kg of cannabis with a street value of up to # 135,000 . lock-up " \
                  ": police raided ahmed \'s premises in bolton after noticing components sold by his online firm " \
                  "permanent discounts were suspiciously clean and undamaged . passing sentence judge peter " \
                  "davies told ahmed : ` it was a sophisticated commercial dismantling of vehicles . you were " \
                  "in charge of an aladdin \'s cave of stolen car parts . \' a second man asim ahmad , 32 , of " \
                  "college drive , whalley range , manchester was jailed for three and a half years after being" \
                  " found guilty of production of cannabis . det con claire waring of greater manchester police " \
                  "said after the case : ` both ahmed and ahmad were involved in this large scale production of " \
                  "cannabis , which would have been supplied in the local area and across greater manchester . ` " \
                  "the judge described the stolen vehicle operation as a ` sophisticated commercial dismantling of" \
                  " vehicles \' , with the value of these vehicles to their owners being in the region of # 130,000 " \
                  ". \' i would encourage anyone who has any information about drugs and other crime in their area " \
                  "to contact police so we can act upon it , as we have done in this case . \'"
        abstract_sentences = [
            ' ramzan ahmed , 34 , sentenced to four years in prison . ',
            ' father-of-three admitted cannabis production and handling stolen goods . ',
            ' officers found wheels , engines and other car parts in his bolton warehouses . ',
            ' also discovered cannabis factory containing 323 plants worth # 135,000 . '
        ]
        vocab = data.Vocab('./data/finished_files/vocab', 10000)
        ex = Example(article, abstract_sentences, vocab)
    """
    def __init__(self, article, abstract_sentences, vocab):
        """
        :param article: raw article string
        :param abstract_sentences: list of raw abstract sentences
        :param vocab: a vocabulary object
        """
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > config.max_enc_steps:
            article_words = article_words[:config.max_enc_steps]

        self.enc_len = len(article_words)
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # Process the abstract
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_tgt_seqs(abs_ids, config.max_dec_steps,
                                                                start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            self.enc_input_extend_vocab, self.article_oov = data.article2ids(article_words, vocab)
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oov)
            _, self.target = self.get_dec_inp_tgt_seqs(abs_ids_extend_vocab, config.max_dec_steps,
                                                       start_decoding, stop_decoding)

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sentences = abstract_sentences

    @staticmethod
    def get_dec_inp_tgt_seqs(sequence, max_len, start_id, stop_id):
        """
        :param sequence: a list of abstract word index
        :param max_len: max decode length
        :param start_id: start of sentence index
        :param stop_id: end of sentence index
        :return: decoder input and decoder output index
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:              # truncate
            inp = inp[:max_len]
            target = target[:max_len]       # no end_token
        else:                               # no truncation
            target.append(stop_id)          # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_tgt(self, max_len, pad_id):
        """
        :param max_len: max decode length within a batch
        :param pad_id: pad index
        :return: padded decoder input and output index
        """
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """
        :param max_len: max decode length within a batch
        :param pad_id: pad index
        :return: padded encoder input index
        """
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    """
    usage: a wrapper of Example class

    example:
        ex = Example(article, abstract_sentences, vocab)
        example_list = [ex for _ in range(10)]
        batch = Batch(example_list, vocab, 10)
    """
    def __init__(self, example_list, vocab, batch_size):
        """
        :param example_list: a list of Example object
        :param vocab: a vocabulary object
        :param batch_size: batch size in training
        """
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)

        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros(self.batch_size, dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        if config.pointer_gen:
            self.max_art_oov = max([len(ex.article_oov) for ex in example_list])
            self.art_oov = [ex.article_oov for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

        for ex in example_list:
            ex.pad_decoder_inp_tgt(config.max_dec_steps, self.pad_id)

        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros(self.batch_size, dtype=np.int32)

        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]
        self.original_abstracts_sentences = [ex.original_abstract_sentences for ex in example_list]


class Batcher(object):
    """
    usage: a wrapper of Batch class
    """
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
            self._finished_reading = False
        else:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1

        self._example_q_threads = []

        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
            self._batch_q_threads = []

        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: {}, '
                'Input queue size: {}'.format(
                    self._batch_queue.qsize(), self._example_queue.qsize())
            )
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None
        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                article, abstract = next(input_gen)
                article, abstract = article.decode(), abstract.decode()
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in data.abstract2sentences(abstract)]
            example = Example(article, abstract_sentences, self._vocab)  # Process into an Example.
            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    @staticmethod
    def text_generator(example_generator):
        while True:
            e = next(example_generator)  # e is a tf.Example
            try:
                article_text = e.features.feature['article'].bytes_list.value[
                    0]  # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[
                    0]  # the abstract text was saved under the key 'abstract' in the data files
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0 or len(abstract_text) == 0:
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            else:
                yield (article_text, abstract_text)
