

import nltk,string
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize) #, stop_words=stopwords.words('russian')
def cosine_sim(text1, text2): #####
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


print('zxc228')
print('результат:')
sentence1='Мясо гигантских тараканов станет вкусной и недорогой альтернативой говядине'
sentence2='Гигантское мясо тараканов станет говядине недорогой и вкусной альтернативой'
print(cosine_sim(sentence1, sentence2))
print(cosine_sim('На столе стояла чашка чая', 'На столе стояла емкость с чаем'))









f1 = open ('C:/Users/Isusi/PycharmProjects/pythonProject/sentence1.txt', 'r', encoding='utf-8')
sentence1 = f1.read()
f1.close()
'''sentence1=('В мировом кинематографе есть ряд фильмов, признанных как критиками, так и зрителями. '
           'Среди них «Побег из Шоушенка». '
           'С августа 2008 г. он неизменно занимает 1-е место в рейтинге «250 лучших фильмов по версии IMDb». '
           'В чём же секрет его успеха?'
           '«Побег из Шоушенка» был снят в 1994 году по рассказу Стивена Кинга '
           '«Рита Хэйворт и искупление в тюрьме Шоушенк». '
           'История, на первый взгляд, начинается довольно банально: '
           'вице-президента солидного банка обвиняют в двойном убийстве, '
           'которого он не совершал, и сажают в тюрьму. '
           'Это был бы самый обычный фильм о тюремной жизни, '
           'если бы не отношение к жизни главного героя Энди Дюфрейна – '
           'даже находясь в заключении, он продолжал ощущать себя свободным человеком.'
           'Каково представление об американской тюрьме, '
           'об отбывании наказания на условиях пожизненных сроков? '
           'Что мы знаем о чувствах людей, виновных и невиновных, которые осознают, '
           'что здесь и умрут? Ничего. А они живут и умирают за решёткой. '
           'Многие находятся там настолько давно, что уже не помнят, какая она, '
           'жизнь на свободе. Здесь, в Шоушенке, свои законы, '
           'а начальник тюрьмы в каком-то смысле законодатель. Но есть и законы неписанные, человеческие.'
           'Повествование ведётся от лица чернокожего заключённого, назвавшего себя при знакомстве с Энди Дюфрейном Рэдом. '
           'Он рассказывает о том, какой путь пришлось пройти главному герою, '
           'прежде чем он стал рабочим тюремной библиотеки, '
           'а впоследствии доверенным лицом начальника тюрьмы по экономическим и юридическим вопросам.'
           'Энди Дюфрейн провёл в тюрьме Шоушенк 20 лет за преступление, которого он не совершал, '
           'и только жажда жизни помогла ему не сломаться, не пасть духом и однажды снова стать свободным.'
           'Не будет преувеличением тот факт, что своим примером Энди Дюфрейн преобразил жизнь других '
           'заключённых: своим умением держаться, просветительством '
           '(он добился расширения библиотечного фонда, давал уроки). '
           'Мы видим, как изменилось отношение к жизни Рэда. '
           'Вероятно, не только его одного. Вспомним хотя бы эпизод, '
           'где Энди Дюфрейн включает на всю колонию аудиотрансляцию дивного оперного пения '
           'на итальянском языке. Как при этом изменились лица заключённых – '
           'из привычно суровых и отрешённых они стали одухотворёнными: '
           'люди услышали голос из мира свободы, в который некоторым не суждено вернуться. '
           'В связи с этим хочется вспомнить, как главный герой говорит о том, '
           'что отнять у человека можно всё, что угодно, кроме того, что находится в душе, '
           'в сердце, в памяти. Никто не может лишить человека его жизненного опыта и воспоминаний, '
           'его убеждений и мировоззрения. '
           'Это очень важный смысловой эпизод, определяющий философию жизни Энди Дюфрейна.')'''

f2 = open ('C:/Users/Isusi/PycharmProjects/pythonProject/sentence2.txt', 'r', encoding='utf-8')
sentence2 = f2.read()
f2.close()

'''sentence2=('Во всем мире есть множество фильмов, которые признают и зрители, и критики. '
           'Одним из них является «Побег из Шоушенка». '
           'С 2008 г. он всегда занимает 1-е место в рейтинге «250 лучших фильмов по версии IMDb». '
           '«Побег из Шоушенка» вышел на экраны в 1994 году. '
           'Он снят по мотивам рассказа Стивена Кинга «Рита Хэйворт и искупление в тюрьме Шоушенк». '
           'История начинается с того, что вице-президента солидного банка обвиняют в двойном убийстве, '
           'которого он не совершал. После этого его сажают в тюрьму. '
           'Но главный герой Энди Дюфрейн даже находясь в заключении не сдается и продолжает '
           'ощущать себя в тюрьме свободным человеком.Какими являются представления у людей об '
           'американской тюрьме и об отбывании наказания на условиях пожизненных сроков? '
           'Что нам известно о людях, виновных и невиновных, которые осознают, '
           'что здесь им суждено умереть? Многие находятся в тюрьме настолько давно, что даже не помнят, '
           'какой является жизнь на свободе. Здесь, в Шоушенке, свои законы, а начальник тюрьмы является '
           'законодателем. Повествование происходит от лица чернокожего заключённого, которого зовут Рэдом. '
           'Он рассказывает о том, какой путь прошел главный герой, '
           'прежде чем он стал рабочим тюремной библиотеки, а впоследствии доверенным лицом начальника тюрьмы.'
           'Энди Дюфрейн провёл в тюрьме 20 лет за преступление, которого он не совершал. '
           'Но его жажда жизни и свободы помогла ему не сломиться, не пасть духом и однажды снова '
           'выйти на свободу.Своим примером Энди Дюфрейн преобразил жизнь других заключённых. '
           'Мы видим, как благодаря ему, изменилось отношение к жизни Рэда. Вероятно, не только его одного. '
           'Можно вспомнить эпизод, где Энди Дюфрейн включает на всю тюрьму аудиотрансляцию оперного '
           'пения на итальянском языке. И благодаря этому изменились лица заключённых – '
           'из суровых и отрешённых они стали одухотворёнными. '
           'Люди услышали голос из мира свободы, в который многим из них уже не получится вернуться. '
           'В связи с этим вспоминается эпизод, в котором главный герой говорит о том, '
           'что отнять у человека можно всё, кроме того, что находится в душе, в сердце, в памяти. '
           'Что никто не может лишить человека жизненного опыта, воспоминаний, его убеждений и мировоззрения. '
           'Это один из важнейших смысловых эпизодов фильма, который определяет философию жизни Энди Дюфрейна.')'''



f3 = open ('C:/Users/Isusi/PycharmProjects/pythonProject/sentence3.txt', 'r', encoding='utf-8')
sentence3 = f3.read()
f3.close()

'''
sentence3=('В международном кино снято множество фильмов,' 
'получивших признание как критиков, так и зрителей.' 
'«Побег из Шоушенка» — один из пунктов списка.' 
'С августа 2008 года он сохраняет свою позицию '
'на первом месте в рейтинге 250 лучших фильмов '
'по версии IMDb. '
'Как ему удалось добиться такого большого успеха?'

'«Побег из Шоушенка» был снят в 1994 году '
'по мотивам книги Стивена Кинга '
'«Рита Хейворт и побег из Шоушенка». '
'Поначалу сюжет начинается внезапно:' 
'вице-президенту известного банка предъявлено обвинение в двойном убийстве, за которое, как он признал, '
'он не был признан виновным, '
'что приводит к длительному тюремному заключению. Свободолюбивое отношение Энди Дюфрена во время его пребывания в тюрьме' 
'сделало этот фильм нетипичным, '
'поскольку вызвало у зрителей чувство '
'неуверенности в его способности выйти на свободу.'

'Каков принцип американских тюрем, '
'например, отбывание пожизненного заключения? '
'Как понимают эмоции личности, '
'как виновные, так и невиновные, '
'осознающие, что это место их неизбежной гибели? '
'Я не знаю, что с этим делать. '
'Они остаются в тюрьме в ожидании неминуемой смерти. '
'Многие прожили так долго, что не могут вспомнить, '
'каково было быть свободным на родине. '
'В Шоушенке действуют законы, '
'отличные от законов города, и начальник тюрьмы в некоторой степени считается законодателем. '
'Существуют и неписаные внутренние человеческие законы.'

'Своим примером Энди Дюфрен '
'изменил жизнь других заключенных, '
'демонстрируя позитивное поведение и '
'проводя занятия (ему удалось увеличить '
'вместимость библиотеки и провести обучение). '
'Красный меняет его подход к жизни. '
'Возможно, что не только он. '
'Давайте не будем забывать эпизод, '
'в котором Энди Дюфрен организовал аудиотрансляцию потрясающего оперного пения на '
'итальянском языке для всей колонии, '
'и он заслужил место. '
'Лица узников из безжалостных '
'и бесчувственных превратились в одухотворенные: в них звучали голоса из мира свободы, голоса невозврата тех, '
'кто всегда был заперт.'

'Хочется вспомнить слова главного '
'героя о том, что можно лишить кого-то всех знаний, к'
'роме души, сердца и памяти. '
'Людям не позволено лишать себя своих убеждений, '
'мировоззрения, жизненного опыта, воспоминаний. '
'Этот смысловой эпизод является решающим моментом в определении жизненной философии Энди Дюфрена.'
)
'''




f4 = open ('C:/Users/Isusi/PycharmProjects/pythonProject/sentence4.txt', 'r', encoding='utf-8')
sentence4 = f4.read()
f4.close()



'''
sentence4=('Иосиф Сталин — выдающийся политик-революционер '
'в истории Российской империи и Советского Союза. '
'Личность и биография главы СССР в обществе '
'по-прежнему громко обсуждаются: '
'одни считают его великим правителем, '
'приведшим страну к победе в '
'Великой Отечественной войне, '
'другие обвиняют в геноциде народа и голодоморе, терроре и насилии над людьми.'
'Родился Иосиф Виссарионович Сталин '
'(настоящая фамилия Джугашвили) 21 декабря '
'1879 года в грузинском городке Гори в семье, принадлежащей к низшему сословию. По другой версии, день рождения будущего '
'вождя пришелся на 18 декабря 1878 года.'
'Он был третьим, но единственным выжившим ребенком в семье — его старшие брат и сестра умерли еще в младенчестве. '
'Сосо, как называла мать будущего правителя СССР, '
'родился не совсем здоровым ребенком, он имел врожденные дефекты конечностей — у него были сросшиеся два пальца на левой ноге, '
'а также усеянные оспинами кожные покровы лица и спины. '
'В раннем детстве с мальчиком случился несчастный случай — '
'его сбил фаэтон, в результате чего у него нарушилась работа левой руки.'
'Отец Иосифа, сапожник по профессии, '
'выпивал, поэтому есть мнение, что он избивал '
'сына и жену. Но в интервью Эмилю Людвигу Сталин отмечал, '
'что родители обращались с ним «неплохо». '
'Когда будущий правитель был еще ребенком, глава семьи погиб, '
'и его дальнейшим воспитанием занималась мать, окружившая сына бесконечной любовью.'
'Изнемогая на трудной работе, '
'желая заработать как можно больше денег на '
'воспитание мальчика, Екатерина Георгиевна старалась '
'вырастить достойного человека, который '
'должен был стать священником. С этой целью '
'она отдала сына в Горийское православное училище, '
'а затем в Тифлисскую духовную семинарию.'
)

'''

sentence5=('Иосиф Сталин является одним из важнейших '
'политиков-революционеров в истории Российской империи '
'и Советского Союза. Биография и личность '
'Сталина до сих пор громко обсуждаются: кто-то считает его '
'великим правителем, приведшим страну к победе в '
'Великой Отечественной войне, а кто-то обвиняет '
'его в геноциде народа, терроре и насилии над людьми.'
'Иосиф Виссарионович Сталин (настоящая фамилия Джугашвили) '
'родился 21 декабря 1879 года в грузинском городке '
'Гори в семье, принадлежащей к низшему сословию. '
'Он был третьим ребенком, родившимся в семье, '
'но единственным выжившим, так как его старшие '
'брат и сестра умерли еще в младенчестве. '
'Сосо, как называла его мать, родился не совсем '
'здоровым ребенком так как у него были врожденные '
'дефекты конечностей — сросшиеся два пальца '
'на левой ноге, а также были покрыты оспинами '
'кожные покровы лица и спины. В раннем детстве '
'с мальчиком случился несчастный случай — '
'его сбил фаэтон. После данного происшествия '
'у него нарушилась работа левой руки.'
'Отец Иосифа был сапожником по профессии. '
'Он часто выпивал, поэтому есть мнение, что '
'он избивал жену и сына. Но в интервью Эмилю '
'Людвигу Сталин отмечал, что родители обращались '
'с ним «неплохо». Когда будущий правитель был '
'еще ребенком, его отец погиб, и дальнейшим '
'воспитанием занималась мать.'
'Изнемогая на трудной работе, желая заработать '
'как можно больше денег на воспитание мальчика, '
'Екатерина Георгиевна хотела вырастить из '
'Иосифа достойного человека, который должен '
'был стать священником. С этой целью она сначала '
'отдала его в Горийское православное училище, '
'а после в Тифлисскую духовную семинарию.'

)


sentence6=(
'И в Российской империи, и в Советском Союзе Иосиф '
'Сталин является исключительным революционным политиком,'
'известным своим исключительным вкладом. '
'Несмотря на репутацию великого лидера, '
'победившего в Великой Отечественной войне, '
'личность и биография главы СССР остаются предметом '
'острых дискуссий в обществе: одни считают его '
'героическим лидером, преодолевшим всеобщие лишения, '
'другие возлагают на него ответственность за геноцид народа '
'и голод, террор и насилие над людьми.'

'Иосиф Виссарионович Сталин (Джугашвили), потомок '
'низшего сословия, родился в Гори, Грузия, 21 декабря 1879 года '
'в семье низшего социального положения. '
'По альтернативной версии, день рождения будущего вождя'
'приходится на 18 декабря 1878 года.'

'Единственным выжившим ребенком в семье был он, '
'третий ребенок в семье, так как его старшие братья и сестры '
'умерли в младенчестве, когда он был еще младенцем. '
'Сосо, мать будущего правителя СССР, не родилась '
'здоровым ребенком из-за врожденных дефектов конечностей, '
'в том числе двух сросшихся пальцев на левой ноге, '
'а также усеянной оспинами кожи лица и спины. '
'В первые годы жизни с мальчиком произошел несчастный '
'случай - на него сбил фаэтон, в результате чего у него '
'ослабла функция левой руки.'

'Отец Иосифа, который работал сапожником, '
'также находился в состоянии алкогольного опьянения, '
'поэтому считается, что он избил своего сына и жену. '
'Эмиль Людвиг попросил Эмиля Сталина рассказать '
'об обращении его родителей со Сталиным, который'
'утверждал, что родители относились к нему «неплохо». '
'В период, когда будущий монарх был еще ребенком, '
'глава семьи вынужден был уйти из жизни, и его '
'дальнейшим воспитанием занималась мать, расточавшая '
'к нему свою любовь бесконечными часами.'

'Из-за изнуренных усилий на востребованной работе '
'Екатерина Георгиевна стремилась заработать достаточный '
'доход, чтобы прокормить свою семью во время воспитания '
'мальчика, поэтому она много работала, чтобы '
'вырастить достойного человека, который станет '
'священником. Именно по этому мотиву она отдала '
'сына в Горийское православное училище, '
'а впоследствии в Тифлисскую духовную семинарию.'

)


print('Кино:')
print(cosine_sim(sentence1, sentence2))

print('Кино2:')
print(cosine_sim(sentence1, sentence3))

print('Биография:')
print(cosine_sim(sentence4, sentence5))

print('Биография2:')
print(cosine_sim(sentence4, sentence6))


'''
print('Не кино:')
print(cosine_sim('я буду яблоко и грушу', 'я буду яблоко или грушу'))

'''


'''

import spacy
from spacy.lang.ru.examples import sentences
nlp = spacy.load("ru_core_news_sm")
doc = nlp("Сегодня я пошел в школу")
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)
    
'''

'''import IPython        #######
from spacy import displacy
#displacy.render(doc[:11], style='dep', jupyter=False)
displacy.serve(doc, style="dep")'''
#http: // localhost: 5000 /

'''nlp = spacy.load("en_core_web_sm")
doc = nlp("While Samsung has expanded overseas, South Korea is still host to most of its factories and research engineers.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)'''

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd




import nltk
import spacy
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem

def lemmat(sentence):
    m = Mystem()
    lemmatized = m.lemmatize(sentence)

    for i in lemmatized:
        if(i==' '):
            lemmatized.remove(i)
    #print(lemmatized) вывод лематизированного текста
    return lemmatized


def normal(sentence):
    stop_words = set(stopwords.words('russian'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = []

    for w in word_tokens:
        #if w not in stop_words:
        filtered_sentence.append(w)
    return filtered_sentence







print("кино3")
'''
separator = ' '

sentence1 = separator.join(normal(sentence1))

sentence1 = separator.join(lemmat(sentence1))


sentence2 = separator.join(normal(sentence2))

sentence2 = separator.join(lemmat(sentence2))


sentence3 = separator.join(normal(sentence3))

sentence3 = separator.join(lemmat(sentence3))



sentence4 = separator.join(normal(sentence4))

sentence4 = separator.join(lemmat(sentence4))


sentence5 = separator.join(normal(sentence5))

sentence5 = separator.join(lemmat(sentence5))

sentence6 = separator.join(normal(sentence6))

sentence6 = separator.join(lemmat(sentence6))


doc1 = nlp(sentence1)
doc2 = nlp(sentence2)
doc3 = nlp(sentence3)
doc4 = nlp(sentence4)
doc5= nlp(sentence5)
doc6= nlp(sentence6)

similarity = doc1.similarity(doc2)
print('результат:')
print(similarity)

similarity = doc1.similarity(doc3)
print('результат2:')
print(similarity)

similarity = doc2.similarity(doc3)
print('результат2:')
print(similarity)


print('Сталин:')
similarity = doc4.similarity(doc5)
print('результат2:')
print(similarity)

print('Сталин:')
similarity = doc5.similarity(doc6)
print('результат2:')
print(similarity)

print('Сталин:')
similarity = doc1.similarity(doc5)
print('результат2:')
print(similarity)
'''


print("zxc2222228888888")

import spacy
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Инициализация
nlp = spacy.load("ru_core_news_lg")
mystem = Mystem()
russian_stopwords = set(stopwords.words("russian"))

def preprocess_text(text):
    # Удаление стоп-слов и лемматизация
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords and token.isalpha()]
    lemmas = mystem.lemmatize(" ".join(tokens))
    return " ".join(lemma.strip() for lemma in lemmas if lemma.strip())




# Предобработка
processed1 = preprocess_text(sentence1)
processed2 = preprocess_text(sentence2)
processed3 = preprocess_text(sentence3)
processed4 = preprocess_text(sentence4)
processed5 = preprocess_text(sentence5)
processed6 = preprocess_text(sentence6)

# Сравнение
doc1 = nlp(processed1)
doc2 = nlp(processed2)
doc3 = nlp(processed3)
doc4 = nlp(processed4)
doc5 = nlp(processed5)
doc6 = nlp(processed6)


similarity = doc1.similarity(doc2)
print(f"Схожесть: {similarity:.4f}")

similarity = doc1.similarity(doc3)
print(f"Схожесть: {similarity:.4f}")

similarity = doc1.similarity(doc4)
print(f"Схожесть: {similarity:.4f}")

similarity = doc1.similarity(doc5)
print(f"Схожесть: {similarity:.4f}")

similarity = doc1.similarity(doc6)
print(f"Схожесть: {similarity:.4f}")












'''
import IPython        #######
from spacy import displacy
#displacy.render(doc[:11], style='dep', jupyter=False)
displacy.serve(doc1, style="dep")
#http: // localhost: 5000 /

'''
def similarity(self, other):
    if 'similarity' in self.doc.user_token_hooks:
        return self.doc.user_token_hooks['similarity'](self)
    if self.vector_norm == 0 or other.vector_norm == 0:
        return 0.0
    return numpy.dot(self.vector, other.vector) / (self.vector_norm * other.vector_norm)












import tkinter as tk
'''
def on_click1():
    label.config(text="Привет, " + str(cosine_sim(sentence1, sentence2)))

def on_click2():
    label.config(text="Привет, " + str(doc1.similarity(doc5)))

app = tk.Tk()
app.title("Мое приложение")
app.geometry("800x600")  # Ширина = 800px, Высота = 600px

label = tk.Label(app, text="Введите имя:")
label.pack()

entry = tk.Entry(app)
entry.pack()

button = tk.Button(app, text="tfidf", command=on_click1)
button.pack()

button2 = tk.Button(app, text="Клик2!", command=on_click2)
button2.pack()

#app.mainloop()
'''





from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Загрузка модели
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Тексты в разных переменных
text1 = sentence1
text2 = sentence2
text3 = sentence3
text4 = sentence4

# Кодирование текстов в векторы
vector1 = model.encode(text1)
vector2 = model.encode(text2)
vector3 = model.encode(text3)
vector4 = model.encode(text4)

# Сравнение text1 с остальными текстами
print("Схожесть " +  str(cosine_similarity([vector1], [vector2])[0][0]))
print("Схожесть " +  str(cosine_similarity([vector1], [vector3])[0][0]))
print("Схожесть " +  str(cosine_similarity([vector1], [vector4])[0][0]))

# Сравнение text2 с text3 (пример дополнительной проверки)
#print(f"\nДополнительно: '{text2}' и '{text3}': {cosine_similarity([vector2], [vector3])[0][0]:.4f}")







import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

selected_file1 = None
selected_file2 = None

sentence1 = ''
sentence2 = ''


def select_file1():
    global selected_file1
    global sentence1
    filepath = filedialog.askopenfilename(
        title="Выберите первый текстовый файл",
        filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
    )
    if filepath:
        selected_file1 = filepath
        label_status1.config(text=f"Файл 1: {selected_file1.split('/')[-1]}")


        f1 = open(selected_file1, 'r', encoding='utf-8')
        sentence1 = f1.read()
        f1.close()


    else:
        label_status1.config(text="Файл 1 не выбран")


def select_file2():
    """Функция для выбора второго файла"""
    global selected_file2
    global sentence2

    filepath = filedialog.askopenfilename(
        title="Выберите второй текстовый файл",
        filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
    )
    if filepath:
        selected_file2 = filepath
        label_status2.config(text=f"Файл 2: {selected_file2.split('/')[-1]}")


        f2 = open(selected_file2, 'r', encoding='utf-8')
        sentence2 = f2.read()
        f2.close()


    else:
        label_status2.config(text="Файл 2 не выбран")


def open_selected_file(file_num):
    selected_file = selected_file1 if file_num == 1 else selected_file2
    if not selected_file:
        messagebox.showwarning("Ошибка", f"Сначала выберите файл {file_num}!")
        return

    try:
        with open(selected_file, "r", encoding="utf-8") as file:
            content = file.read()

        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, f"Содержимое файла {file_num}:\n\n{content}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось открыть файл {file_num}:\n{e}")


def compare1():
    global sentence1
    global sentence2
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, str(cosine_sim(sentence1, sentence2)))


def compare2():

    global sentence1
    global sentence2

    processed1 = preprocess_text(sentence1)
    processed2 = preprocess_text(sentence2)
    doc1 = nlp(processed1)
    doc2 = nlp(processed2)

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, str(doc1.similarity(doc2)))



def compare3():

    global sentence1
    global sentence2

    text1 = sentence1
    text2 = sentence2


    vector1 = model.encode(text1)
    vector2 = model.encode(text2)



    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, str(cosine_similarity([vector1], [vector2])[0][0]))


# Создаем основное окно
app = tk.Tk()
app.title("Работа с двумя файлами")
app.geometry("750x600")

# Фрейм для кнопок выбора файлов
select_frame = tk.Frame(app)
select_frame.pack(pady=10)

# Кнопка для выбора первого файла
btn_select1 = tk.Button(
    select_frame,
    text="Выбрать файл 1",
    command=select_file1,
    padx=10,
    pady=5,
    bg="#2196F3",
    fg="white"
)
btn_select1.pack(side=tk.LEFT, padx=5)

# Кнопка для выбора второго файла
btn_select2 = tk.Button(
    select_frame,
    text="Выбрать файл 2",
    command=select_file2,
    padx=10,
    pady=5,
    bg="#FF9800",
    fg="white"
)
btn_select2.pack(side=tk.LEFT, padx=5)

# Метки для отображения статуса
label_status1 = tk.Label(app, text="Файл 1 не выбран", font=("Arial", 10))
label_status1.pack()
label_status2 = tk.Label(app, text="Файл 2 не выбран", font=("Arial", 10))
label_status2.pack()

# Фрейм для кнопок открытия файлов
open_frame = tk.Frame(app)
open_frame.pack(pady=10)

# Кнопка для открытия первого файла
btn_open1 = tk.Button(
    open_frame,
    text="Открыть файл 1",
    command=lambda: open_selected_file(1),
    padx=10,
    pady=5,
    bg="#4CAF50",
    fg="white"
)
btn_open1.pack(side=tk.LEFT, padx=5)

# Кнопка для открытия второго файла
btn_open2 = tk.Button(
    open_frame,
    text="Открыть файл 2",
    command=lambda: open_selected_file(2),
    padx=10,
    pady=5,
    bg="#9C27B0",
    fg="white"
)
btn_open2.pack(side=tk.LEFT, padx=5)

# Поле для вывода текста (с прокруткой)
text_area = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=80, height=20)
text_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Фрейм для кнопок сравнения и результата
compare_frame = tk.Frame(app)
compare_frame.pack(pady=10, fill=tk.X, padx=10)


btn_compare1 = tk.Button(
    compare_frame,
    text="Tf-idf",
    command=compare1,
    padx=15,
    pady=5,
    bg="#607D8B",
    fg="white"
)
btn_compare1.pack(side=tk.LEFT, padx=5)

# Кнопка Сравнить2

btn_compare2 = tk.Button(
    compare_frame,
    text="Word2vec",
    command=compare2,
    padx=15,
    pady=5,
    bg="#795548",
    fg="white"
)
btn_compare2.pack(side=tk.LEFT, padx=5)



btn_compare3 = tk.Button(
    compare_frame,
    text="Sentence-BERT",
    command=compare3,
    padx=15,
    pady=5,
    bg="#795548",
    fg="white"
)
btn_compare3.pack(side=tk.LEFT, padx=5)



# Текстовое поле для вывода результата сравнения
result_text = tk.Text(
    compare_frame,
    wrap=tk.WORD,
    width=40,
    height=3,
    bg="#F5F5F5",
    padx=5,
    pady=5
)
result_text.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

app.mainloop()