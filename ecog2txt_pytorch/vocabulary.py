
class Vocabulary:
    def __init__(self, file_path):
        self.load_from_file(file_path)

    def load_from_file(self, file_path):
        with open(file_path, 'r') as f:
            x = f.readlines()
            self.words_ind_map = {x[1].strip(): x[0] for x in enumerate(x)}
            self.ind_words_map = {x[0]: x[1].strip() for x in enumerate(x)}