import pandas as pd
#tokenizer = Tokenizer(num_words=5000)


df_test = pd.read_csv('../../../measuring-customer-happiness/test_hp.csv', encoding='utf-8')
#print(df_test)
df_test = df_test['Description']
df_test.head()
for i in df_test:
    i = tokenizer.texts_to_sequences(i)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

    model.predict(instance)