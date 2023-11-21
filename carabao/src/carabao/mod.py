#===============================================================================
# module: mod.py
#===============================================================================
"""
This module supplies a single function reverse_words that reverses
a string word by word.

>>> reverse_words('four score and seven years')
'years seven and score four'

>>> reverse_words('1')
'1'

You must call reverse_words with one single argument, a string:

>>> reverse_words()
Traceback (most recent call last):
    ...
TypeError: reverse_words() missing 1 required positional argument: 'astring'
"""
def reverse_words(astring):
    words = astring.split()
    words.reverse()
    return ' '.join(words)

#===============================================================================
# doctest
#===============================================================================

if __name__ == '__main__':
    import doctest            # to run doctest: $ python mod.py
    doctest.testmod()         #             or: $ python mod.py -v
