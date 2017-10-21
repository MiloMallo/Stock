import os
import pandas as pd
from cnn import CNN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

random.seed(42)

class TrainCNN:

    def __init__(self, num_historical_days, days=10, pct_change=0):
        self.data = []
        self.labels = []
        self.test_data = []
        self.test_labels = []
        self.cnn = CNN(num_features=5, num_historical_days=num_historical_days, is_train=False)
        files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
        for file in files:
            print(file)
            df = pd.read_csv(file, index_col='Date', parse_dates=True)
            df = df[['Open','High','Low','Close','Volume']]
            labels = df.Close.pct_change(days).map(lambda x: [int(x > pct_change/100.0), int(x <= pct_change/100.0)])
            df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
            df['labels'] = labels
            df = df.dropna()
            test_df = df[:365]
            df = df[400:]
            data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            labels = df['labels'].values
            for i in range(num_historical_days, len(df), num_historical_days):
                self.data.append(data[i-num_historical_days:i])
                self.labels.append(labels[i-1])
            data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            labels = test_df['labels'].values
            for i in range(num_historical_days, len(test_df), 1):
                self.test_data.append(data[i-num_historical_days:i])
                self.test_labels.append(labels[i-1])



    def random_batch(self, batch_size=128):
        batch = []
        labels = []
        data = zip(self.data, self.labels)
        i = 0
        while True:
            i+= 1
            while True:
                d = random.choice(data)
                if(d[1][0]== int(i%2)):
                    break
            batch.append(d[0])
            labels.append(d[1])
            if (len(batch) == batch_size):
                yield batch, labels
                batch = []
                labels = []

    def train(self, print_steps=100, display_steps=100, save_steps=1000, batch_size=128, keep_prob=0.6):
        if not os.path.exists('./cnn_models'):
            os.makedirs('./cnn_models')
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        if os.path.exists('./logs/train'):
            for file in [os.path.join('./logs/train/', f) for f in os.listdir('./logs/train/')]:
                os.remove(file)
        if os.path.exists('./logs/test'):
            for file in [os.path.join('./logs/test/', f) for f in os.listdir('./logs/test')]:
                os.remove(file)

        sess = tf.Session()
        loss = 0
        l2_loss = 0
        accuracy = 0
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('./logs/train')
        test_writer = tf.summary.FileWriter('./logs/test')
        sess.run(tf.global_variables_initializer())
        if os.path.exists('./cnn_models/checkpoint'):
            with open('./cnn_models/checkpoint', 'rb') as f:
                model_name = next(f).split('"')[1]
            #saver.restore(sess, "./models/{}".format(model_name))
        for i, [X, y] in enumerate(self.random_batch(batch_size)):
            _, loss_curr, accuracy_curr = sess.run([self.cnn.optimizer, self.cnn.loss, self.cnn.accuracy], feed_dict=
                    {self.cnn.X:X, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
            loss += loss_curr
            accuracy += accuracy_curr
            if (i+1) % print_steps == 0:
                print('Step={} loss={}, accuracy={}'.format(i, loss/print_steps, accuracy/print_steps))
                loss = 0
                l2_loss = 0
                accuracy = 0
                test_loss, test_accuracy, confusion_matrix = sess.run([self.cnn.loss, self.cnn.accuracy, self.cnn.confusion_matrix], feed_dict={self.cnn.X:self.test_data, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                print("Test loss = {}, Test accuracy = {}".format(test_loss, test_accuracy))
                print(confusion_matrix)
            if (i+1) % save_steps == 0:
                saver.save(sess, './cnn_models/cnn.ckpt', i)

            if (i+1) % display_steps == 0:
                summary = sess.run(self.cnn.summary, feed_dict=
                    {self.cnn.X:X, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
                train_writer.add_summary(summary, i)
                summary = sess.run(self.cnn.summary, feed_dict={
                    self.cnn.X:self.test_data, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                test_writer.add_summary(summary, i)


if __name__ == '__main__':
    cnn = TrainCNN(num_historical_days=20, days=10, pct_change=10)
    cnn.train()
