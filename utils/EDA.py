import random
import nltk
from nltk.corpus import wordnet
import torch
from torch import nn


class EDA(nn.Module):
    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1):
        super().__init__()
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()

                if " " in synonym or synonym == word:
                    continue
                synonyms.add(synonym)


        return list(synonyms)

    def synonym_replacement(self, words, n):
        new_words = words[:]
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for word in random_word_list:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return new_words

    def random_insertion(self, words, n):
        new_words = words[:]
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1 and counter < 10:
            word = random.choice(words)
            synonyms = self.get_synonyms(word)
            counter += 1
        if synonyms:
            insert_syn = random.choice(synonyms)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_syn)

    def random_swap(self, words, n):
        new_words = words[:]
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, words):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return words

    def random_deletion(self, words, p):
        if len(words) == 1:
            return words
        return [word for word in words if random.random() > p]

    def forward(self, sentences):
        if isinstance(sentences, torch.Tensor):

            raise ValueError("Input must be a list of strings")

        outputs = []
        for sentence in sentences:
            words = sentence.split()
            num_words = len(words)

            n_sr = max(1, int(self.alpha_sr * num_words))
            n_ri = max(1, int(self.alpha_ri * num_words))
            n_rs = max(1, int(self.alpha_rs * num_words))

            augmented = self.synonym_replacement(words, n_sr)

            outputs.append(" ".join(augmented))

        return outputs

if __name__ == "__main__":
    sentence = "a photo of a cat"
    text_augment = EDA(alpha_sr=0.5)
    print(text_augment([sentence]))
