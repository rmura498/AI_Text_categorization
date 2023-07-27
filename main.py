import os.path
import tarfile
import pandas as pd
from extracting_data import compile_data, compile_dictionary
from cleaning_dataset import removing_stop_words, stemming

if not os.path.exists('./dataset'):
    file = tarfile.open('reuters21578.tar.gz')
    file.extractall('./dataset')
    file.close()

dataset = compile_data()
dataset_dicts = [compile_dictionary(data) for data in dataset]

# print(dataset_dicts[0])
dataset_dicts = [data for data in dataset_dicts if data['TOPICS'] != 'none' or data['REUTERS TOPICS'] == 'YES']
for data in dataset_dicts:
    if data['BODY'] == '':
        data['BODY'] = data['TITLE']

df = pd.DataFrame(dataset_dicts)
df = removing_stop_words(df)
print(df['BODY'])
df = stemming(df)
print(df['BODY'])
df.to_csv('./cleaned_csv')









# 0 eliminare stop word
# 0.1 eliminare termini senza capacità discriminante (articoli e preposizioni)
# 0.2 eseguire lemmantizzazione ridurre ogni parola alla sua redice (make, making, made -> mak)
# https://pypi.org/project/PorterStemmer/
# TODO
# 1. feature selection
# 1.1 valutare la capacità discriminante di ciascun termine nel training set, selezionare gli N termini più discriminanti
# con N da definire, e rappresentare ogni documento con un vettore di occorrenze, frequenze o altro (per esempio TF-IDF sez 5.1 eq 1)
# degli N termini selezionati
# 1.2 per valutare la capacità discriminante si può usare una delle misure definite nella sezione 5.4 e.g inform gain.

# 2. addestrare M classificatori one vs all (le 5 classi con più documenti)
# 2.1 individuare le 5 classi


# 3 valutazione performances con F_\beta combinazione di precision e recall
# 3.1 usare versione macro.averaged, sezione 7 pag 36
