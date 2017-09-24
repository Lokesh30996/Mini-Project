import pandas
data=pandas.read_csv('D:/Fall Sem 17-18/Mini Project/stanfordSentimentTreebank_Dataset/stanfordSentimentTreebank/datasetSentences.csv')
category=pandas.read_csv('D:/Fall Sem 17-18/Mini Project/stanfordSentimentTreebank_Dataset/stanfordSentimentTreebank/datasetSplit.csv')
labels=pandas.read_csv('D:/Fall Sem 17-18/Mini Project/stanfordSentimentTreebank_Dataset/stanfordSentimentTreebank/sentiment_labels.csv')
split= pandas.merge(data, category, on='sentence_index')
split_data= pandas.merge(split, labels, on='sentence_index')
train_split=split_data.loc[split_data['splitset_label']==1]
dev_split=split_data.loc[split_data['splitset_label']==3]
test_split=split_data.loc[split_data['splitset_label']==2]
train=train_split[['sentence']].copy()
train_sentiment_values=train_split[['sentiment values']].copy()
test=test_split[['sentence']].copy()
test_sentiment_values=test_split[['sentiment values']].copy()
dev=dev_split[['sentence']].copy()
dev_sentiment_values=dev_split[['sentiment values']].copy()
test.to_csv('test.csv')
dev.to_csv('dev.csv')
train.to_csv('train.csv')
test_sentiment_values.to_csv('test_sentiment_values.csv')
train_sentiment_values.to_csv('train_sentiment_values.csv')
dev_sentiment_values.to_csv('dev_sentiment_values.csv')