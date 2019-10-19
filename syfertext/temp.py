text = ' I  love apples very much  '

# The number of characters in the text
text_size = len(text)

# Initialize a pointer to the position of the first character of 'text'
pos = 0

# This is a flag to indicate whether the character we are comparing
# to is a white space or not
is_space = text[0].isspace()

# Start tokenization
for i, char in enumerate(text):


    # We are looking for a character that is the opposit of 'is_space'
    # if 'is_space' is True, then we want to find a character that is
    # not a space. and vice versa. This event marks the end of a token.
    is_current_space = char.isspace()
    if  is_current_space != is_space:

        print(text[pos:i])

        if is_current_space:
            pos = i + 1
        else:
            pos = i

        # Update the character type of which we are searching
        # the opposite (space vs. not space).
        # prevent 'pos' from being out of bound
        if pos < text_size:
            is_space = text[pos].isspace()                

    # Create the last token if the end of the string is reached
    if i == text_size - 1 and pos <= i:
        print(text[pos:])
