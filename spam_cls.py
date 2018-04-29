import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from nltk import SnowballStemmer, re, downloader
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
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


def strip_url(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        text = text.replace(url, '_URL_')
    text = text.replace('https:', '')
    return text


def convert_emphesize(text):
    emphs = re.findall(r'\b[A-Z]{2,}\b', text)
    emphs = set(emphs)
    for emph_ in emphs:
        text = re.sub(r'\b' + emph_ + r'\b', emph_ + ' emphh', text)
    return text


def trivial_tokenizer(text):
    return text


def is_long_number(text, threshold=5):
    numbers_lens = re.findall('\\d+', text)
    if numbers_lens and len(max(numbers_lens, key=len)) >= threshold:
        return text + ' _longnumber_'
    return text


max_features = 800
test_size = .2

# Import data
current_path = os.getcwd()
train = pd.read_csv(r"C:\Users\Lior\Downloads\InterviewHomeworkTask "
                    r"2018\InterviewHomeworkTask\interviewClassificationTask.csv", encoding='ISO-8859-1')

fields = ['v1', 'v2_concat']

x_train_ = train[fields[1]].fillna("fillna").values.tolist()  # Make sure no cell stays empty
y_train = train[fields[0]].values

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

input_vectorizer = CountVectorizer(tokenizer=trivial_tokenizer, max_features=max_features, lowercase=False)
tfidf = TfidfTransformer()
digits_counter = FunctionTransformer(validate=False)
linSVC = LinearSVC()
pipeline = [('Vectorizer', input_vectorizer), ('TFIDF', tfidf), ('LinSVC', linSVC)]
model = Pipeline(pipeline)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print(classification_report(y_test, y_predicted))
