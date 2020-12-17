import re
from collections import Counter
import matplotlib.pyplot as plt

text = "red pink pink blue blue yellow ORANGE BLUE BLUE PINK"

text_lowercase = text.lower()

words = re.findall(r'\w+', text_lowercase)

print(words)


# create vocabulary
vocab = set(words)

# add information with word count
counts_a = dict()
for w in words:
    counts_a[w] = counts_a.get(w, 0) + 1

print(counts_a)

# create vocab with collections.Counter
counts_b = dict()
counts_b = Counter(words)
print(counts_b)