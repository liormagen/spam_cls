from collections import Counter

import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from nltk import SnowballStemmer, re, downloader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import statistics
import os

filters = '!"#%&()*+,-./:;<=>?_@[\\]^`{|}~\t\n0123456789'


def clean_sw():
    try:
        sw = stopwords.words('english')
    except LookupError:
        downloader.download('stopwords')
        sw = stopwords.words('english')
    return set([english_stemmer(w) for w in sw])


def english_stemmer(word):
    stemmed_word = SnowballStemmer('english').stem(word)
    return stemmed_word


def strip_url(text, return_count=False):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if return_count:
        return len(urls)
    for url in urls:
        text = text.replace(url, '_URL_')
    text = text.replace('https:', '')
    return text


def convert_emphesize(text, return_count=False):
    emphs = re.findall(r'\b[A-Z]{2,}\b', text)
    emphs = set(emphs)
    if return_count:
        return len(emphs)
    for emph_ in emphs:
        text = re.sub(r'\b' + emph_ + r'\b', emph_ + ' emphh', text)
    return text


def trivial_tokenizer(text):
    return text


def is_long_number(text, threshold=5, flag_res=False):
    numbers_lens = re.findall('\\d+', text)
    if numbers_lens and len(max(numbers_lens, key=len)) >= threshold:
        if flag_res:
            return True
        return text + ' _longnumber_'
    if flag_res:
        return False
    return text


max_features = 800
test_size = .2

# Import data
current_path = os.getcwd()
train = pd.read_csv(r"D:\Source Code\spam_cls\data\interviewClassificationTask.csv", encoding='ISO-8859-1')

fields = ['v1', 'v2_concat']

x_train_ = train[fields[1]].fillna("fillna").values.tolist()  # Make sure no cell stays empty
y_train = train[fields[0]].values

# Showing some data characteristics
class_convert = {1: ['Spam'], 0: ['Not Spam']}
number_hist = Counter()
emph_counter = {'emph': [], 'non emph': []}
spam_emph, non_spam_emph = [], []
spam_urls, non_spam_urls = [], []
for x_, y_ in zip(x_train_, y_train):
    if is_long_number(x_, flag_res=True):
        number_hist.update(class_convert[int(y_)])
    emph_count = convert_emphesize(x_, return_count=True)
    if emph_count:
        # ratio_ = round(len(x_.split(' ')) / emph_count)
        ratio_ = emph_count
        if (y_ == [0]).all():
            non_spam_emph.append(ratio_)
        else:
            spam_emph.append(ratio_)
    # if strip_url(x_, return_count=True) > 0:
    if (y_ == [0]).all():
        non_spam_urls.append(strip_url(x_, return_count=True))
    else:
        spam_urls.append(strip_url(x_, return_count=True))

print('Number of occurrences of long numbers in non-spam sentences: %s' % number_hist[0])
print('Number of occurrences of long numbers in spam sentences: %s' % number_hist[1])

print('Trying to prove that spam messages contain more emphasized words than non-spam messages')
fig1 = plt.figure(1)
plt.hist(spam_emph)
plt.title('Spam emphasized words histogram - avg=%.3f, std=%.3f' % (statistics.mean(spam_emph),
                                                                          statistics.stdev(spam_emph)))
plt.xlabel('Ratio')
plt.ylabel('Count')
fig1.show()

fig2 = plt.figure(2)
plt.title('Not-spam emphasized words histogram - avg=%.3f, std=%.3f' % (statistics.mean(non_spam_emph),
                                                                              statistics.stdev(non_spam_emph)))
plt.hist(non_spam_emph)
plt.xlabel('Ratio')
plt.ylabel('Count')
fig2.show()

fig3 = plt.figure(3)
plt.hist(spam_emph, label='Spam')
plt.hist(non_spam_emph, label='Not spam')
plt.legend(loc='upper right')
plt.title('Emphasized words count - Spam VS not-spam')
fig3.show()

fig4 = plt.figure(3)
plt.hist(spam_urls, label='Spam')
plt.hist(non_spam_urls, label='Not spam')
plt.legend(loc='upper right')
plt.title('Number of URLs in text - Spam VS not-spam')
fig4.show()

# Showing that in spam messages there's a dominant amount of URLs
# class_convert = {1: ['Spam'], 0: ['Not Spam']}
# number_hist = Counter()
# emph_counter = {'url exist': [], 'no url': []}
# spam_urls, non_spam_urls = [], []
# for x_, y_ in zip(x_train_, y_train):
#     if strip_url(x_, return_count=True):
#         number_hist.update(class_convert[int(y_)])
#     emph_count = convert_emphesize(x_, return_count=True)
#     if emph_count:
#         # ratio_ = round(len(x_.split(' ')) / emph_count)
#         ratio_ = emph_count
#         if (y_ == [0]).all():
#             non_spam_emph.append(ratio_)
#         else:
#             spam_emph.append(ratio_)


# Pre Processing data
lengths = 0
stem_it = True
sw = clean_sw()
# Clean repeating chars - looooooooooooooooooove -> love
pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

for idx, doc in enumerate(x_train_):
    doc = strip_url(doc)
    doc = is_long_number(doc)
    doc = pattern.sub(r"\1", doc)
    doc = convert_emphesize(doc)
    tokens = [english_stemmer(w) for w in text_to_word_sequence(doc, filters=filters, lower=True)]
    x_train_[idx] = [w for w in tokens if w not in sw]
    lengths += len(x_train_[idx])

max_len = round(lengths / idx)
print('Average length: %s' % max_len)

x_train, x_test, y_train, y_test = train_test_split(x_train_, y_train,
                                                    test_size=test_size)

# I'll use the vectorizer while keeping upper case
input_vectorizer = CountVectorizer(tokenizer=trivial_tokenizer, max_features=max_features, lowercase=False)
tfidf = TfidfTransformer()
digits_counter = FunctionTransformer(validate=False)
linSVC = LinearSVC()
pipeline = [('Vectorizer', input_vectorizer), ('TFIDF', tfidf), ('LinSVC', linSVC)]
model = Pipeline(pipeline)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print(classification_report(y_test, y_predicted))
