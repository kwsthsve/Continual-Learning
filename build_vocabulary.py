import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

nltk.download('punkt')


class Vocabulary(object):

    def __init__(self,
                 vocab_file,
                 annotations_file=None,
                 vocab_threshold=5,
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 pad_word="<pad>",
                 vocab_from_file=False):

        """
        Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          pad_word: Special word denoting padding token.
          annotations_file: Path for train annotation file.
        """

        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.annotations_file = annotations_file
        self.counter = Counter()
        self.vocab_file = vocab_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):

        # Load the vocabulary from file OR build the vocabulary from scratch.
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
                self.idx = vocab.idx
                self.counter = vocab.counter
            print(f'\nVocabulary successfully loaded from {self.vocab_file} file!\n')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):

        # Populate the dictionaries for converting tokens to integers (and vice-versa).
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        self.add_captions()

    def init_vocab(self):

        # Initialize the dictionaries for converting tokens to integers (and vice-versa).
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):

        # Add a token to the vocabulary.
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self, annotations=None):

        # In case of full dataset training, task training and GEM: annotations == None means it is the (first) task encountered so we build vocabulary from self.annotations file
        if annotations == None:
            annotations = self.annotations_file

        # Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold.
        coco = COCO(annotations)
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            self.counter.update(tokens)

        words = [word for word, cnt in self.counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

        # Save vocabulary
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self, f)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
