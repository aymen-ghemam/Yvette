import os

import pandas as pd
from sklearn.model_selection import train_test_split

from shutil import copyfile, copy

data = {}

for label in os.listdir('data'):
    data[label] = []
    for image in os.listdir(os.path.join('data', label)):
        data[label].append(image)


train_data = pd.DataFrame(columns=['names', 'labels'])
validation_data = pd.DataFrame(columns=['names', 'labels'])
test_data = pd.DataFrame(columns=['names', 'labels'])


for label in data:
    images = data[label]
    x_train, x_validate, y_train, y_validate = train_test_split(images, [label]*len(images), test_size = 0.2, random_state = 12345)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 12345)

    train = [[x_train[i], y_train[i]] for i in range(len(x_train))]
    validate = [[x_validate[i], y_validate[i]] for i in range(len(x_validate))]
    test = [[x_test[i], y_test[i]] for i in range(len(x_test))]

    train_data = pd.concat([train_data, pd.DataFrame(data=train ,columns=['names', 'labels'])])
    validation_data = pd.concat([validation_data, pd.DataFrame(data=validate ,columns=['names', 'labels'])])
    test_data = pd.concat([test_data, pd.DataFrame(data=test ,columns=['names', 'labels'])])

train_data.to_csv('train.csv')
validation_data.to_csv('validate.csv')
test_data.to_csv('test.csv')

import os
from shutil import copyfile, copy


train_path = pd.read_csv('train.csv')
validate_path = pd.read_csv('validate.csv')
test_path = pd.read_csv('test.csv')

labels = train_path['labels'].unique()

if not os.path.isdir('train/cop'):
    for label in labels:
        os.makedirs(os.path.join('train',label))

    for label in labels:
        os.makedirs(os.path.join('test',label))

    for label in labels:
        os.makedirs(os.path.join('validate',label))

for i in range(len(train_path)):
    image_path = train_path.loc[i, 'names']
    copyfile(os.path.join('data', train_path.loc[i,'labels'], image_path), os.path.join('train', train_path.loc[i,'labels'], '{}-{}.jpg'.format(train_path.loc[i,'labels'], i)))

for i in range(len(validate_path)):
    image_path = validate_path.loc[i, 'names']
    copyfile(os.path.join('data', validate_path.loc[i,'labels'], image_path), os.path.join('validate', validate_path.loc[i,'labels'], '{}-{}.jpg'.format(validate_path.loc[i,'labels'], i)))

for i in range(len(test_path)):
    image_path = test_path.loc[i, 'names']
    copyfile(os.path.join('data', test_path.loc[i,'labels'], image_path), os.path.join('test', test_path.loc[i,'labels'], '{}-{}.jpg'.format(test_path.loc[i,'labels'], i)))
