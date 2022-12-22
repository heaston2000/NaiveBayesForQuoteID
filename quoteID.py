"""
Hugh Easton - Middlebury College - CS 311
November 2022

THIS IS A MODEL FOR PREDICTING WHO IN A BOOK SAID A QUOTE

This script defines the QuoteNB model, which is takes in a dictionary mapping a set of speakers to quotes
and creates a Naive Bayes model, for predicting who said a given quote in a book.
It also takes us through the testing process of that model towards the end, and outputs some nifty
vizualizations allowing us to see what words make a character's speech unique.

"""
import LoadQuotes
from typing import List, Sequence#, Tuple # DefaultDict, Generator, Hashable, Iterable, 
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class QuoteNB:
    """Naive Bayes model for predicting quote origin on a dictionaryy of paired speakers and quotes"""
    
    def __init__(self, _speakers):
        """
        Args:
            speakers: Dictionaries of speakers paired with each of the quotes that each said
        """
        
        
        self.speakers = _speakers
        self.speaker_totals =  defaultdict(int) # Total quotes that each person said 
        self.num_speakers = len(self.speakers) # The total number of speakers present:
        
        self.word_totals = defaultdict(int) # The total number of words each character has said
        self.speaker_word_dict = defaultdict(lambda: defaultdict(int)) # Nested dictionary w/ key speaker value dictionary w/ key word, value count of instances of that word
        self.speaker_unique_dict = defaultdict(int) # Number of unique words each speaker has said
        
        # Now the variables for holding information about bigrams
        self.bigram_dict = defaultdict(lambda: defaultdict(int)) # Nested dictionary of the number of occurences of each bigram across each speaker
        self.bigram_totals = defaultdict(int) # holds the total bigrams each speaker has said
        self.unique_bigram_dict = defaultdict(int) # the number of unique bigrams each speaker has said
        
        # Now the variables for holding information about trigrams
        self.trigram_dict = defaultdict(lambda: defaultdict(int)) # Dictionary of the number of occurences of each trigram across each speaker
        self.trigram_totals = defaultdict(int) # holds the total trigrams each speaker has said
        self.unique_trigram_dict = defaultdict(int) # the number of unique trigrams each speaker has said
        
        # Now the variables for holding information about average word length 
        self.len_increments = [3,5.5] # (small < 3 < medium < 4.5 < large)
        self.len_types = ["S", "M", "L"]
        self.word_len_dict = defaultdict(lambda: defaultdict(int)) # Nested Dictionary of the amount of times speaker has said a sentence with a small medium and large
        # The number of quotes a speaker has said is already accounted for in self.speaker_totals
        self.unique_len_dict = defaultdict(int) # should end up being the amount of categories that the speaker's totals end up in (i.e. 3 for most cases)
        
        # Now the variables for holding information about number of apostrophes 
        self.apos_increments = [1, 2, 5] # (small = 0 < medium < 2 < large < 5 < XL)
        self.apos_types = ["S", "M", "L", "XL"]
        self.apos_dict = defaultdict(lambda: defaultdict(int)) # Nested dictionary of the amount of times each speaker has had sentence w/ each amount of apostrophes
        # The number of quotes a speaker has said is already accounted for in self.speaker_totals
        self.unique_apos_dict = defaultdict(int) # Should end up being of length 4 the amount of apostrophe segments
        
        # Now the variables for holding information about number of periods 
        self.period_increments = [1, 2, 5] # (small = 0 < medium < 2 < large < 5 < XL)
        self.period_types = ["S", "M", "L", "XL"]
        self.period_dict = defaultdict(lambda: defaultdict(int)) # Nested dictionary of the amount of times each speaker has had sentence w/ each amount of periods
        # The number of quotes a speaker has said is already accounted for in self.speaker_totals
        self.unique_period_dict = defaultdict(int) # Should end up being of length 4 the amount of period segments
        
        # Now the variables for holding information about 4-grams
        self.four_dict = defaultdict(lambda: defaultdict(int)) # Dictionary of the number of occurences of each 4-gram across each speaker
        self.four_totals = defaultdict(int)
        self.unique_four_dict = defaultdict(int)
        
        
        # Now load in the quotes and create the model:
        self.add_examples(self.speakers)
        
        pass


    def preprocess(self, example) -> List[str]:
        """Normalize the string into a list of words. i.e. take out all punctuation,
        return lower case words in a  list format.

        Arguments:
            example (str): Text input to split and normalize
        Returns:
            List[str]: Normalized words
        """
        example = example.lower() # Make example all lower case
        i = 0
        while i < (len(example)): # iterate through each character in the example, removing if it is punctuation
            current_letter = example[i]
            if not (current_letter.isalpha() or current_letter.isnumeric()) and current_letter != ' ':
                example = example[:i] + '' + example[(i+1):] # Remove the character if it is not a number or letter or space
            else:
                i = i + 1 # Otherwise continue
                
        return example.split() # split into list of the words that were separated by spaces

    def add_examples(self, speaker_dict):
        """Add all training examples which come in the form of a dictionary, to the model

        Args:
            speaker_dict: dictionary of each speaker paired with quotes that the speaker said
        """
        
        for speaker in speaker_dict.keys():
            self.speaker_totals[speaker] = len(speaker_dict[speaker]) # Count the amount of times this speaker shows up
            for quote in speaker_dict[speaker]: # For each quote this speaker has been identified as saying
                apos_count = quote.count("'") # count the number of apostrophes that show up
                for i in range(len(self.apos_increments)):
                    # For each of the size increments of apostrophe categories, if we have less apostrophes
                    #   than that size increment, then we identify it as that category
                    if apos_count < self.apos_increments[i]:
                        apos_category = self.apos_types[i]
                        break
                    else:
                        apos_category = self.apos_types[len(self.apos_increments)]
                # If this speaker hasn't said a quote with that category of apostrophes before, increment the unique categories
                #   of apostrophes, either way increment this category of apostrophe sizes for this speaker
                if self.apos_dict[speaker][apos_category] == 0:
                    self.unique_apos_dict[speaker] += 1
                self.apos_dict[speaker][apos_category] += 1
                
                period_count = quote.count(".") # count the number of period that show up
                for i in range(len(self.period_increments)):
                    # For each of the size increments of apostrophe categories, if we have less periods
                    #   than that size increment, then we identify it as that category
                    if period_count < self.period_increments[i]:
                        period_category = self.period_types[i]
                        break
                    else:
                        period_category = self.period_types[len(self.period_increments)]
                # If this speaker hasn't said a quote with that category of period before, increment the unique categories
                #   of apostrophes, either way increment this category of period sizes for this speaker
                if self.period_dict[speaker][period_category] == 0:
                    self.unique_period_dict[speaker] += 1
                self.period_dict[speaker][period_category] += 1
                    
                
                word_list = self.preprocess(quote) # Remove punctuation + make quote all lowercase
                word_len_sum = 0
                for word in word_list:
                    word_len_sum += len(word)
                    # Add to the total amount of words the speaker has said:
                    self.word_totals[speaker] += 1
                    # Add to the total amount of times we've seen this word in particular/if the word is unique increment the amount of unique words
                    #   seen
                    if self.speaker_word_dict[speaker][word] == 0:
                        self.speaker_unique_dict[speaker] += 1
                    self.speaker_word_dict[speaker][word] += 1
                    
                    # Now for 4-grams
                    if len(word) >= 4:
                        for j in range(len(word)-4):
                            curr_4gram = word[j:(j+4)]
                            self.four_totals[speaker] += 1
                            if self.four_dict[speaker][curr_4gram] == 0:
                                self.unique_four_dict[speaker] += 1
                            self.four_dict[speaker][curr_4gram] += 1
                    
                # Calculate average word length and update corresponding counter variables:
                avg_word_len = word_len_sum / len(word_list)
                for i in range(len(self.len_increments)):
                    if avg_word_len < self.len_increments[i]:
                        len_category = self.len_types[i] # get what this length category is called ex: "small"
                        if self.word_len_dict[speaker][len_category] == 0:
                            self.unique_len_dict[speaker] += 1
                        self.word_len_dict[speaker][len_category] += 1
                        break
                    else: # this will consistently call the word the biggest length increment, and will be the last thing assigned to len_category
                        len_category = self.len_types[len(self.len_increments)]
                # If the category is the largest word:
                if len_category == self.len_types[len(self.len_increments)]:
                    if self.word_len_dict[speaker][len_category] == 0:
                        self.unique_len_dict[speaker] += 1
                    self.word_len_dict[speaker][len_category] += 1

                # Count the bigrams:
                for i in range(len(word_list)-1):
                    curr_bigram = word_list[i] + ' ' + word_list[i+1] # bigrams are stored as each word with a space in between them
                    self.bigram_totals[speaker] += 1
                    if self.bigram_dict[speaker][curr_bigram] == 0:
                        self.unique_bigram_dict[speaker] += 1
                    self.bigram_dict[speaker][curr_bigram] += 1
                # Count the trigrams NOT USED:
                for i in range(len(word_list)-2):
                    curr_trigram = word_list[i] + ' ' + word_list[i+1] + ' ' + word_list[i+2] 
                    self.trigram_totals[speaker] += 1
                    if self.trigram_dict[speaker][curr_trigram] == 0:
                        self.unique_trigram_dict[speaker] += 1
                    self.trigram_dict[speaker][curr_trigram] += 1
                
        
        pass

    def predict(self, example, pseudo=0.000001) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Arguments:
            example: string, quote to predict the speaker of 
            pseudo: float, Pseudo-count for Laplace smoothing. Defaults to 0.0001.

        Returns:
            p_speaker_given_quote, a dictionary with speakers as keys, each value is the probability that that speaker said this quote
        """

        # Start by finding the category of apostrophes this falls in:        
        apos_count = example.count("'")
        for i in range(len(self.apos_increments)):
            if apos_count < self.apos_increments[i]:
                apos_cat = self.apos_types[i]
                break
            else:
                apos_cat = self.apos_types[len(self.apos_increments)]     
        # and for periods:
        period_count = example.count(".")
        for i in range(len(self.period_increments)):
            if period_count < self.period_increments[i]:
                period_cat = self.period_types[i]
                break
            else:
                period_cat = self.period_types[len(self.period_increments)]
        
        
        word_list = self.preprocess(example) # List of the words w/o punctuation that the speaker says in this quote
        
        normalizing_sum = np.log(0) # The denominator the sum of all probabilities P(words | speaker) for all speakers
        
        p_speaker_given_quote = defaultdict(float) # This will eventually hold the probabilities for all the speakers
        
        for speaker in self.speakers:
            prob_this_speaker = self.speaker_totals[speaker] / self.num_speakers
            # Add unconditional probability that this speaker is speaking:
            numerator_this_speaker = [prob_this_speaker]
            
            # Add conditional probs P(apostrophes | this speaker) and P(periods | this speaker)
            numerator_this_speaker.append((self.apos_dict[speaker][apos_cat] + pseudo)/(self.speaker_totals[speaker] + (self.unique_apos_dict[speaker] * pseudo)))
            numerator_this_speaker.append((self.period_dict[speaker][period_cat] + pseudo)/(self.speaker_totals[speaker] + (self.unique_period_dict[speaker] * pseudo)))

            # Add the conditional probabilities P(word | this_speaker) to the numerator factors
            word_len_total = 0
            for word in word_list:
                numerator_this_speaker.append((self.speaker_word_dict[speaker][word] + pseudo)/(self.word_totals[speaker] + (self.speaker_unique_dict[speaker] * pseudo)))
                word_len_total += len(word)
                
                if len(word) >= 4:
                    for j in range(len(word) - 4):
                        fourgram = word[j:(j+4)]
                        numerator_this_speaker.append((self.four_dict[speaker][fourgram] + pseudo)/(self.four_totals[speaker] + (self.unique_four_dict[speaker] * pseudo)))
                
            # Add probability P(word_len | speaker)
            average_word_len = word_len_total / len(word_list)
            for i in range(len(self.len_increments)):
                if average_word_len < self.len_increments[i]:
                    len_cat = self.len_types[i] # what category of average word length this falls under
                    break
                else:
                    len_cat = self.len_types[len(self.len_increments)]
            numerator_this_speaker.append((self.word_len_dict[speaker][len_cat] + pseudo)/(self.speaker_totals[speaker] + (self.unique_len_dict[speaker] * pseudo)))
            
            
            # Now count bigrams and update numerator:
            for i in range(len(word_list)-1):
                bigram = word_list[i] + ' ' + word_list[i+1]
                #print(self.bigram_totals[speaker])
                #print(self.unique_bigram_dict[speaker])
                if (self.bigram_totals[speaker] + (self.unique_bigram_dict[speaker] * pseudo)) != 0: # to fix a bug
                    numerator_this_speaker.append((self.bigram_dict[speaker][bigram] + pseudo)/(self.bigram_totals[speaker] + (self.unique_bigram_dict[speaker] * pseudo)))
                else:
                    numerator_this_speaker.append(1)
#             # Now count trigrams and update numerator
#             for i in range(len(word_list)-2):
#                 trigram = word_list[i] + ' ' + word_list[i+1] + ' ' + word_list[i+2]
#                 numerator_this_speaker.append((self.trigram_dict[speaker][trigram] + pseudo)/(self.trigram_totals[speaker] + (self.unique_trigram_dict[speaker] * pseudo)))
# 
            # Multiply the whole sum of all the conditional probabilities together by putting it into log space
            log_sum_numerator = np.sum(np.log(numerator_this_speaker))
            
            p_speaker_given_quote[speaker] = log_sum_numerator
        
        # Divide by the normalizing factor which is the sum across all speakers given this quote
        normalizing_sum = np.logaddexp.reduce(list(p_speaker_given_quote.values()))
        for speaker in self.speakers:
            p_speaker_given_quote[speaker] = np.exp(p_speaker_given_quote[speaker] - normalizing_sum) # needs to be done in log space to prevent underflow
        
        return p_speaker_given_quote
    
    
    def FeatureBarChartWords(self, character):
        """
        Given the character/speaker, prints a bar chart of the top features for this character as well as probabilities for each feature
        
        Arguments:
        - character: string, the speaker we want the bar chart for
        """
        probabilities = np.array([])
        words = np.array([])
        for word in self.speaker_word_dict[character]:
            pseudo = 0.000001
            words = np.append(words, word) # to keep track of what word corresponds with what probability
            # prob_initial = P(word | speaker) for the speaker we are concerned with
            prob_initial = (self.speaker_totals[character] /self.num_speakers) * (self.speaker_word_dict[character][word] + pseudo)/(self.word_totals[character] + (self.speaker_unique_dict[character] * pseudo))
            normalizer = 0
            # Create a normalizer, summing across all the other speakers of the P(word | speaker)
            for speaker in self.speakers:
                normalizer += (self.speaker_totals[speaker] / self.num_speakers) * (self.speaker_word_dict[speaker][word] + pseudo)/(self.word_totals[speaker] + (self.speaker_unique_dict[speaker] * pseudo))
            probabilities = np.append(probabilities, (prob_initial / normalizer))

        topfiveind = np.argpartition(probabilities, -5)[-5:] # find the indices of the highest probabilities
        topfiveind = topfiveind[np.argsort(-probabilities[topfiveind])] # sort theindices from highest to lowest probability they represent
        
        # Now create the bar plot demonstrating the top words for P(character | word)
        plt.bar(words[topfiveind], probabilities[topfiveind], align='center', alpha=0.5)
        plt.ylim(.99999, 1)
        label_y = 'P(speaker is "' + character +  '" | word)'
        plt.ylabel(label_y)
        plt.yticks([1])
        plt.xlabel("Word")
        title = 'Words most uniquely identified with speaker "' + character + '"'
        plt.title(title)
        plt.show()
    
def Test(speaker_dict, model):
    """
    TEST the model, i.e. make predictions for each quote and test if each prediction is right
    
    Arguments:
    - speaker_dict: dictionary, mapping speaker to their corresponding quotes
    - model: pre-trained QuoteNB model for quote identification
    Returns:
    - Fraction of quotes correctly identified
    """
    total_quotes  = 0
    total_correct = 0
    for speaker in speaker_dict.keys():
        for quote in speaker_dict[speaker]:
            total_quotes += 1
            speaker_probs = model.predict(quote)
            speaker_pred = max(speaker_probs, key = speaker_probs.get)
            if speaker_pred == speaker:
                total_correct += 1
    return total_correct / total_quotes # accuracy metric is just percent correct
    
def main(story_path, characters, plot_character):
    """
    Load the text and then the dictionary mapping speakers with their respective quotes, then train the model and run some tests!
    
    Arguments:
    - story_path: string, the file path of where the .txt file of the book is located
    - characters: list of strings, each of the character names of this book
    - plot_character: False if we don't want to generate a plot of the words most uniquely ID'ing a character
                        if we do, it should hold a string with the character's name
    """
    
    text = LoadQuotes.LoadScript(story_path) # load and preprocess the book
    speaker_dict = LoadQuotes.FindQuotes(text, characters) # create quote dictionary mapping characters to each of their quotes
    
    # Translate dictionary to a list of tuples, that way we can split it into a test and training set later
    speaker_list = [(quote, key) for key, quotes in speaker_dict.items() for quote in quotes]

    
    test_reps = 10 # number of times we should train and test a model
    test_avg = 0
    train_avg = 0
    for i in range(test_reps):
        # Expects df where each quote is a row
        train_df, test_df = train_test_split(list(speaker_list), test_size=0.2)
        
        #Put the list of speaker, quote pairs (in the training set) back into a dictionary so we can use the model with them
        train_dict = defaultdict(lambda: [])
        for quote_speaker_pair in train_df:
            train_dict[quote_speaker_pair[1]].append(quote_speaker_pair[0])
        #Now do it for the test set:
        test_dict = defaultdict(lambda: [])
        for quote_speaker_pair in test_df:
            test_dict[quote_speaker_pair[1]].append(quote_speaker_pair[0])
            
        # Create the model and test it!
        model = QuoteNB(train_dict)
        test_acc = Test(test_dict, model)
        test_avg += test_acc
        train_acc = Test(train_dict, model)
        train_avg += train_acc
        
    print("Train Average Accuracy: ", train_avg / test_reps)
    print("Test Average Accuracy: ", test_avg / test_reps)
    if plot_character != False: 
        model.FeatureBarChartWords(plot_character)


"""
THE SECTION BELOW IS USED TO INITIATE THE TRAINING AND TESTING OF MODELS
Depending on your value of story, you can choose which book to train a model on!
"""
characters_mobydick = ['ahab', 'i', 'queequeg', 'starbuck', 'stubb', 'tashtego', 'flask', 'daggoo', 'pip', 'fedallah']
characters_jekyllhyde = ['utterson', 'poole', 'hyde', 'jekyll', 'i']
characters_gats = ['gatsby', 'i', 'daisy', 'tom', 'myrtle', 'meyer', 'jordan']
characters_lilwomen = ['jo', 'amy', 'meg', 'beth', 'marmee', 'aunt', 'laurie', 'laurence', 'john']
characters_PnP = ["darcy", "hurst", "elizabeth", "jane", "bingley", "bennet", "wickham", "lydia"]
story = "Dick" # "Jekyll" for Dr. Jekyll and Mr. Hyde, "Dick" for Moby Dick, "Gatsby" for The Great Gatsby, "PnP" for Pride and Prejudice, and "Lilwomen" for Little Women
plot_character = "ahab" # lowercase version of the character's name who you want to see a plot of the most unique words of, False if you don't want such a plot
if story == "Jekyll":
    main("DRJMRH.txt", characters_jekyllhyde, plot_character)
elif story == "Dick":
    main("MOBYDICK.txt", characters_mobydick, plot_character)
elif story == "Gatsby":
    main("GATSBY.txt", characters_gats, plot_character)
elif story == "Lilwomen":
    main("LILWOMEN.txt", characters_lilwomen, plot_character)
elif story == "PnP":
    main("PRIDEPREJUDICE.txt", characters_PnP, plot_character)
else:
    print("Invalid story name!")
    
