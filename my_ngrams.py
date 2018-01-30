import arpa


class NGram(object):
    def lm_word(self, word, lm):
        if word in lm.vocabulary():
            return word
        word = word[:-1]
        if word in lm.vocabulary():
            return word
        return '[UNKNOWN]'


class OneGram(NGram):
    def guess_words(self, probs: list):
        models = arpa.loadf("data/ukn.1.lm")
        lm = models[0]
        alpha = 10.

        scores = []
        guesses = []

        for prob in probs:
            score = {}
            for word in prob:
                word = self.lm_word(word, lm)
                lm_score = lm.log_p(word)
                score[word] = prob[word] + alpha * lm_score

            scores.append(score)

        for score in scores:
            guess = max(score, key=score.get)
            guesses.append(guess)

        return guesses


class TwoGram(NGram):
    def guess_words(self, test_set, probs: list):
        models = arpa.loadf("data/ukn.2.lm")

        lm = models[0]
        sentences = []

        for video_num in test_set.sentences_index:
            sentence_probs = [probs[word_index] for word_index in test_set.sentences_index[video_num]]

            sentence = self._guess_sentence(sentence_probs, lm)
            sentences.extend(sentence)

        return sentences

    def _guess_sentence(self, probs: list, lm):
        probs_begin_end = [{'<s>': 0.}] + probs + [{'</s>': 0.}]
        begin_candidates = [(['<s>'], 0.)]

        breadth = 8
        alpha = 10.
        best_guesses = self._viterbi_path(probs_begin_end, lm, begin_candidates, breadth, alpha)
        # print('guess: {}'.format(best_guesses[0]))
        sentence = best_guesses[0][0]

        return sentence[1:-1]
    
    def _viterbi_path(self, probs: list, lm, candidates: list, breadth: int, alpha: float):
        """
        :param candidates: list of tuples of the form (path, value)
        :param breadth: maximum number of nodes that will be searched in one state
        :param alpha: weight of the language model compared to the score provided by the HMM
        """
        word_count = len(candidates[0][0])
        if word_count >= len(probs):
            return candidates

        prob = probs[word_count]
        new_candidates = []
        for candidate in candidates:
            path = candidate[0]
            value = candidate[1]
            for word in prob:
                words_for_lm = self.lm_word(path[-1], lm) + ' ' + self.lm_word(word, lm)
                ngram_value = lm.log_p(words_for_lm)
                word_prob = prob[word]
                word_value = alpha * ngram_value + word_prob
                new_path = path + [word]
                new_value = value + word_value
                new_candidates.append((new_path, new_value))

        new_candidates.sort(key = lambda tup: tup[1], reverse = True)
        new_candidates = new_candidates[:breadth]

        return self._viterbi_path(probs, lm, new_candidates, breadth, alpha)
    

class ThreeGram(NGram):
    def guess_words(self, test_set, probs: list):
            models = arpa.loadf("data/ukn.3.lm")

            lm = models[0]
            sentences = []

            for video_num in test_set.sentences_index:
                sentence_probs = [probs[word_index] for word_index in test_set.sentences_index[video_num]]

                sentence = self._guess_sentence(sentence_probs, lm)
                sentences.extend(sentence)

            return sentences

    def _guess_sentence(self, probs: list, lm):
        probs_begin_end = [{'<s>': 0.}] + probs + [{'</s>': 0.}]
        begin_candidates = [(['<s>'], 0.)]

        breadth = 8
        alpha = 10.
        best_guesses = self._viterbi_path(probs_begin_end, lm, begin_candidates, breadth, alpha)
        # print('guess: {}'.format(best_guesses[0]))
        sentence = best_guesses[0][0]

        return sentence[1:-1]

    def _viterbi_path(self, probs: list, lm, candidates: list, breadth: int, alpha: float):
        """
        This method calls itself recursively. candidates is a list of tuples of the form (path, value), where path is
        a list of words and value is a probability value for this path. In each recursion step, one word is added to the
        paths. Example: in one step, the elements of candidates have the form

        (['JOHN', 'WRITE'], 2.6356)

        and in the next step, they have the form

        (['JOHN', 'WRITE', 'HOMEWORK'], 1.125)


        :param probs: list of word probabilities as calculated by the recognizers
        :param candidates: list of tuples of the form (path, value)
        :param breadth: maximum number of nodes that will be searched in one state
        :param alpha: weight of the language model compared to the score provided by the HMM
        """
        word_count = len(candidates[0][0])
        if word_count >= len(probs):
            return candidates

        prob = probs[word_count]
        new_candidates = []
        for candidate in candidates:
            path = candidate[0]
            value = candidate[1]
            for word in prob:
                if len(path) == 1:
                    words_for_lm = self.lm_word(path[-1], lm) + ' ' + self.lm_word(word, lm)
                else:
                    words_for_lm = self.lm_word(path[-2], lm) + ' ' + self.lm_word(path[-1], lm) + ' ' + self.lm_word(word, lm)

                ngram_value = lm.log_p(words_for_lm)
                word_prob = prob[word]
                word_value = alpha * ngram_value + word_prob
                new_path = path + [word]
                new_path_value = value + word_value
                new_candidates.append((new_path, new_path_value))

        # Sort the new candidates by their value, and use the <breadth> most promising candidates only for the next
        # recursion step.
        new_candidates.sort(key=lambda tup: tup[1], reverse=True)
        new_candidates = new_candidates[:breadth]

        return self._viterbi_path(probs, lm, new_candidates, breadth, alpha)






