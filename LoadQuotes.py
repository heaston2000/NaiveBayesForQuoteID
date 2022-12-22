"""
Hugh Easton - Middlebury College - CS 311
November 2022

The purpose of this script is to:
 - Load in a story via a .txt file
 - Identify and separate quotes within that story
 - Attribute each quote to a certain speaker
 - Return a dictionary mapping each speaker to their set of quotes
"""
import re
from collections import defaultdict

def LoadScript(file_name):
    """
    Preprocess and load a book, so that it is ready for quote identification
    Arguments:
    - file_name: name of the .txt file the book from Project Gutenberg comes from
    Returns:
    - stor_str_final: The preprocessed version of the book.
    """
    # Load the context of a file as one long string:
    with open(file_name, 'r') as file:
        story_str_long = file.read()

    # Remove new lines EXCEPT for paragraph breaks
    #   (aside: the format of books from project gutenberg has new lines in the middle
    #       of paragraphs to make it easier to read, new paragraphs are identified by
    #       two new lines)
    # First take paragraph breaks out
    story_str_special_newline = story_str_long.replace('\n\n', 'thisisaveryspecialparagraphbreak')
    # and take all new lines out
    story_str_nonewlines = story_str_special_newline.replace('\n', ' ')
    # then put paragraph breaks back in
    story_str_final = story_str_nonewlines.replace('thisisaveryspecialparagraphbreak', '\n')
        
    return story_str_final

def FindQuotes(story_str, characters):
    """
    Purpose: to take in a string of the text of a story and return a mapping of each speaker to their respective quotes
    Arguments:
    - story_str: a string, the text of the story
    - characters: a list of strings, each of the names of the characters who might be speaking in the story
    Returns
    - speaker_dict: a dictionary with keys the names of each speaker in string form and values as a list of the quotes each speaker has said
    """
    # Construct a regular expression of what a quote should look like
    quote_format = re.compile(r'"(.+?)"')
    
    # List of strings, one for each quote
    quotes = quote_format.findall(story_str)
    
    speaker_dict = defaultdict(lambda: []) # This will speakers mapping to each quote a speaker says (to be returned at the end)
    last_speakerL = False # To monitor if the last quote we know who spoke it (on the left side of the quote)
    last_speakerR = False # same but for the right side of the quote
    for quote in quotes:
        
        start_idx = story_str.find(quote)
        end_idx = start_idx + len(quote)

        # First: check the quad of words on either side of the quote:
        right_quad = story_str[end_idx+1:story_str.find(' ', story_str.find(' ' , story_str.find(' ', story_str.find(' ', end_idx)+1)+1)+1)].split()
        left_quad = story_str[story_str.rfind(' ', 0, story_str.rfind(' ' , 0, story_str.rfind(' ', 0, story_str.rfind(' ', 0, start_idx)))):start_idx-1].split()
        
        # First check the right side for a possible speaker:
        speaker_foundR = FindSpeaker(right_quad, characters)
        if speaker_foundR != False:
            speaker_dict[speaker_foundR].append(quote)
            last_speakerR = speaker_foundR
            last_speakerL = False
        else: # Then check the left:
            speaker_foundL = FindSpeaker(left_quad, characters)
            if speaker_foundL != False:
                speaker_dict[speaker_foundL].append(quote)
                last_speakerL = speaker_foundL
                last_speakerR = False
            # If speaker is neither on the left or right sides of the quote, use the last speaker from the right side of the quote (if that existed)
            elif last_speakerR != False:
                speaker_dict[last_speakerR].append(quote)
                last_speakerR = False
                last_speakerL = False
                
        
    return speaker_dict


def FindSpeaker(passage, characters):
    """
    Purpose: Given a passage of text, checks if their is a character's name in that passage. Meant to be used to check if
    a section of string holds the speaker of a given quote
    Arguments:
     - passage: portion of text to be searched as a list of strings of words (punctuation included)
     - characters: list of characters that might be speaking in this text
    """
    for i in range(len(passage)):
        curr_word = Normalize(passage[i])
        if curr_word in characters:
            return curr_word
    return False

def Normalize(str_input):
    """
    Purpose: Given a string, make it all lower case and remove punctuation
    """
    str_lower = str_input.lower() # Make example all lower case
    i = 0
    while i < (len(str_lower)): # iterate through each character in the example, removing if it is punctuation
        current_letter = str_lower[i]
        if current_letter == "'": # If letter is an apostrophe just include everything up to that apostrophe
            str_lower = str_lower[:i]
            break
        if not (current_letter.isalpha() or current_letter.isnumeric()) and current_letter != ' ':
            str_lower = str_lower[:i] + '' + str_lower[(i+1):] # Remove the character if it is not a number or letter or space
        else:
            i = i + 1 # Otherwise continue
    return str_lower
