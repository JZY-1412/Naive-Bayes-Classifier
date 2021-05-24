class NaiveBayesClassifier:

    def __init__(self):
        """
        Unique class is stored as self.unique_class.
        Total number of classes is stored as self.total_classes_number.
        Total number of each class is stored in self.class_number_dict.
        Total number of unique words is stored as self.unique_word_number.
        Total number of a word in a class is stored in self.class_word_number_dict.
        otal number of a word in a class is stored in self.class_total_words_dict.
        """
        self.unique_classes = []
        self.total_classes_number = 0
        self.class_number_dict = {}
        self.unique_word_number = 0
        self.class_word_number_dict = {}
        self.class_total_words_dict = {}

    def fit(self, abstracts, classes):
        classes = np.array(classes)
        abstracts = np.array(abstracts)
        # unique class
        self.unique_classes = sorted(set(classes))

        # total number of classes
        self.total_classes_number = len(classes)

        # total number of each class
        c, counts = np.unique(classes, return_counts=True)
        self.class_number_dict = dict(zip(c, counts))

        # total number of unique words
        all_text = ""
        for text in abstracts:
            all_text = all_text + text + " "
        all_text = all_text.split()
        unique_word = set(all_text)
        self.unique_word_number = len(unique_word)

        # total number of a word in a class
        self.class_word_number_dict = {}
        class_text_dict = {}
        for c in self.unique_classes:
            class_text_dict[c] = ""
        for i in range(len(abstracts)):
            c = classes[i]
            text = abstracts[i]
            class_text_dict[c] = class_text_dict[c] + text + " "
        for c in class_text_dict:
            text = np.array(class_text_dict[c].split())
            word, counts = np.unique(text, return_counts=True)
            self.class_word_number_dict[c] = dict(zip(word, counts))

        # total number of words in a class
        self.class_total_words_dict = {}
        for c in class_text_dict:
            text = class_text_dict[c].split()
            self.class_total_words_dict[c] = len(text)

    def predict(self, abstracts, ids):
        result = []
        for i in range(len(ids)):
            pred_id = ids[i]
            pred_abs = abstracts[i].split()
            id_result_prob = []
            for c in self.unique_classes:
                prob_c = np.log(self.class_number_dict[c] / self.total_classes_number)
                prob_x = 0
                word_number_dict = self.class_word_number_dict[c]
                total_words_number = self.class_total_words_dict[c]
                for word in pred_abs:
                    if word in word_number_dict:
                        word_number = word_number_dict[word]
                    else:
                        word_number = 0
                    prob_word = np.log((word_number + 1) / (total_words_number + self.unique_word_number))
                    prob_x = prob_x + prob_word
                prob = prob_c + prob_x
                id_result_prob.append([c, prob])

            id_result_prob = sorted(id_result_prob, key=lambda x: x[1])
            result.append((pred_id, id_result_prob[-1][0]))

        return result