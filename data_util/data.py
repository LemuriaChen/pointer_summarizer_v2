import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'         # used to pad the encoder input, decoder input and target sequence (index -> 0)
UNKNOWN_TOKEN = '[UNK]'     # used to represent out-of-vocabulary words (index -> 1)
START_DECODING = '[START]'  # used at the start of every decoder input sequence (index -> 2)
STOP_DECODING = '[STOP]'    # used at the end of untruncated target sequences (index -> 3)


class Vocab(object):
    """
    usage: read a preprocessed word frequency files and construct a vocabulary object

    example:
        vocab = Vocab('./data/finished_files/vocab', 10000)
        print(vocab.word2id('the'))
    """
    def __init__(self, vocab_file, max_size):
        """
        :param vocab_file: path of vocabulary
        :param max_size: size of vocabulary
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                line = line.strip('\n')
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: {}'.format(line))
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(r'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, '
                                    r'but {} is'.format(w))
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: {}'.format(w))
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as {}; we now have {} words. Stopping reading.".format(
                        max_size, self._count))
                    break
        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(
            self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: {}'.format(word_id))
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, file_path):
        print("Writing word embedding metadata file to {}".format(file_path))
        with open(file_path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    """
    :param data_path: 'data/finished_files/chunked/*'
    :param single_pass: True or False
    :return:
        an iterable tensorflow.core.example.example_pb2.Example object

    example:
        ex = example_generator('data/finished_files/chunked/*', False)
        print(next(ex))
    """
    while True:
        file_list = glob.glob(data_path)
        assert file_list, ('Error: Empty file list at %s'.format(data_path))
        if single_pass:
            file_list = sorted(file_list)
        else:
            random.shuffle(file_list)
        for f in file_list:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break


def article2ids(article_words, vocab):
    """
    :param article_words: a list of words in a article
    :param vocab: a vocabulary object
    :return: a list of word index in a article, a list of out of vocabulary words in an article
    """
    ids = []
    oov = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)

    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w not in oov:
                oov.append(w)
            oov_num = oov.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oov


def abstract2ids(abstract_words, vocab, article_oov):
    """
    :param abstract_words: a list of words in a abstract
    :param vocab: a vocabulary object
    :param article_oov: a list of out of vocabulary words in the corresponding article
    :return: a list of word index in a abstract
    """
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oov:
                vocab_idx = vocab.size() + article_oov.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def output2words(id_list, vocab, article_oov):
    """
    :param id_list: a list of inferred word index
    :param vocab: a vocabulary object
    :param article_oov: a list of out of vocabulary words in the corresponding article
    :return: inferred abstract
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)
        except ValueError:
            assert article_oov is not None, \
                "Error: model produced a word ID that isn't in the vocabulary. " \
                "This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oov[article_oov_idx]
            except ValueError:
                raise ValueError('Error: model produced word ID {} which corresponds to article OOV {} '
                                 'but this example only has {} article OOVs'.format
                                 (i, article_oov_idx, len(article_oov)))
        words.append(w)
    return words


def abstract2sentences(abstract):
    """
    :param abstract: string of a abstract
    :return: a list of sentence string in a abstract

    example:
        abstract = r'<s> moon will pass in front of sun tomorrow creating 98 % eclipse in some parts . ' \
          r'</s> <s> the event due to take place at 9.30 am , meaning children will be at school . ' \
          r'</s> <s> parents angry after some headteachers decide to keep youngsters indoors .' \
          r' </s> <s> school chiefs say they will not risk children \'s eyesight from looking at sun . </s>'

        print(abstract2sentences(abstract))
    """
    cur = 0
    sentences = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sentences.append(abstract[start_p + len(SENTENCE_START): end_p])
        except ValueError:
            return sentences


def show_art_oov(article, vocab):
    """
    :param article: string of the article
    :param vocab: a vocabulary object
    :return: oov-encoded string of the article
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = ["__{}__".format(w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oov(abstract, vocab, article_oov):
    """
    :param abstract: string of the abstract
    :param vocab: a vocabulary object
    :param article_oov: a list of out of vocabulary words in the corresponding article
    :return: oov-encoded string of the abstract
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:
            if article_oov is None:
                new_words.append("__{}__".format(w))
            else:
                if w in article_oov:
                    new_words.append("__{}__".format(w))
                else:
                    new_words.append("!!__{}__!!".format(w))
        else:
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str

