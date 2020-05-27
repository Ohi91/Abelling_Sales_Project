import pickle
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from label_studio.ml import LabelStudioMLBase


class AbellingModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(AbellingModel, self).__init__(**kwargs)

        # then collect all keys from config which will be used to extract data from task and to form prediction
        # Parsed label config contains only one output of <Choices> type
        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        assert self.info['type'] == 'Choices'

        # the model has only one textual input
        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            # If there is no trainings, define cold-started the simple TF-IDF text classifier
            self.reset_model()
            # This is an array of <Choice> labels
            self.labels = self.info['labels']
            # make some dummy initialization
            self.model.fit(X=self.labels, y=list(range(len(self.labels))))
            self.model2.fit(X=list([[i, i, i, i, i] for i in range(len(self.labels))]), y=list(range(len(self.labels))))
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
        else:
            # otherwise load the model from the latest training results
            self.model_file = self.train_output['model_file']
            self.model_file2 = self.train_output['model_file2']
            with open(self.model_file, mode='rb') as f:
                self.model = pickle.load(f)
            with open(self.model_file2, mode='rb') as f:
                self.model2 = pickle.load(f)
            # and use the labels from training outputs
            self.labels = self.train_output['labels']
            print('Loaded from train output with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name=self.to_name, labels=str(self.labels)
            ))
			
			
    def extract_feature(self, text):
        text = text.split('\n')  
        feat_dict = {}
        text_feat = ''
        neumcat_feat = []

        for i in range(1,len(text)-1):
            pair = text[i].split(': ')
            if pair[1] == '-':
                pair[1] = str(np.nan)

            if pair[0] == 'Categories' or pair[0] == 'Short-Desc' or pair[0] == 'HQ-location':
                text_feat += pair[1] + ' ' 

            elif pair[0] == 'Acquired-By':
                if pair[1] == 'nan':
                    neumcat_feat.append(0)
                else: 
                    neumcat_feat.append(1)

            elif pair[0] == 'Estimated-Revenue':
                if pair[1] == 'nan':
                    neumcat_feat.append(0)
                elif pair[1] == 'Less than $1M':
                    neumcat_feat.append(1)
                elif pair[1] == '$1M to $10M':
                    neumcat_feat.append(2)
                elif pair[1] == '$10M to $50M':
                    neumcat_feat.append(3)
                elif pair[1] == '$50M to $100M':
                    neumcat_feat.append(4)
                elif pair[1] == '$10B+':
                    neumcat_feat.append(5)

            else:
                neumcat_feat.append(int(pair[1]))

        return text_feat, neumcat_feat
		
    def reset_model(self):
        self.model = make_pipeline(TfidfVectorizer(), LogisticRegression(C=10, verbose=True))
        self.model2 = make_pipeline(LogisticRegression(C=10, verbose=True))

    def predict(self, tasks, **kwargs):
        # collect input texts
        input_texts = []
        input_feats = []
        for task in tasks:
            temp = self.extract_feature(task['data'][self.value])
            input_texts.append(temp[0])
            input_feats.append(temp[1])
            print(temp[1])

        # get model predictions
        probabilities = self.model.predict_proba(input_texts)
        probabilities2 = self.model2.predict_proba(input_feats)
        for i in range(len(probabilities)):
            probabilities[i] = ((probabilities[i]+probabilities2[i])/2.0)
        predicted_label_indices = np.argmax(probabilities, axis=1)
        predicted_scores = probabilities[np.arange(len(predicted_label_indices)), predicted_label_indices]
        predictions = []
        for idx, score in zip(predicted_label_indices, predicted_scores):
            predicted_label = self.labels[idx]
            # prediction result for the single task
            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': score})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        input_texts = []
        input_feats = []
        output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}
        for completion in completions:
            # get input text from task data

            if completion['completions'][0].get('skipped'):
                continue

            temp = self.extract_feature(completion['data'][self.value])
            input_texts.append(temp[0])
            input_feats.append(temp[1])
  

            # get an annotation
            output_label = completion['completions'][0]['result'][0]['value']['choices'][0]
            output_labels.append(output_label)
            output_label_idx = label2idx[output_label]
            output_labels_idx.append(output_label_idx)

        new_labels = set(output_labels)
        added_labels = new_labels - set(self.labels)
        if len(added_labels) > 0:
            print('Label set has been changed. Added ones: ' + str(list(added_labels)))
            self.labels = list(sorted(new_labels))
            label2idx = {l: i for i, l in enumerate(self.labels)}
            output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        self.reset_model()
        self.model.fit(input_texts, output_labels_idx)
        self.model2.fit(input_feats, output_labels_idx)

        # save output resources
        model_file = os.path.join(workdir, 'model.pkl')
        with open(model_file, mode='wb') as fout:
            pickle.dump(self.model, fout)
			
        model_file2 = os.path.join(workdir, 'model2.pkl')
        with open(model_file2, mode='wb') as fout:
            pickle.dump(self.model2, fout)

        train_output = {
            'labels': self.labels,
            'model_file': model_file,
			'model_file2': model_file2
        }
        return train_output
