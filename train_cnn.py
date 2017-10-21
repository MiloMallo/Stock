import os
import pandas as pd
from cnn import CNN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"]=""

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
            for i in range(num_historical_days, len(df), num_historical_days):
                self.data.append(df[['Open', 'High', 'Low', 'Close', 'Volume']].values[i-num_historical_days:i])
                self.labels.append(df['labels'].values[i-1])
            for i in range(num_historical_days, len(test_df), 1):
                self.test_data.append(test_df[['Open', 'High', 'Low', 'Close', 'Volume']].values[i-num_historical_days:i])
                self.test_labels.append(test_df['labels'].values[i-1])



    def random_batch(self, batch_size=128):
        batch = []
        labels = []
        data = zip(self.data, self.labels)
        while True:
            batch.append(random.choice(data))
            if (len(batch) == batch_size):
                yield batch, labels
                batch = []
                labels = []

    def train(self, print_steps=100, display_data=100, save_steps=1000):
        if not os.path.exists('./cnn_models'):
            os.makedirs('./cnn_models')
        sess = tf.Session()
        loss = 0
        l2_loss = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        with open('./cnn_models/checkpoint', 'rb') as f:
            model_name = next(f).split('"')[1]
        #saver.restore(sess, "./models/{}".format(model_name))
        for i, [X, y] in enumerate(self.random_batch(self.batch_size)):
           
            _, loss_curr = sess.run([self.cnn.optimizer, self.loss], feed_dict=
                    {self.cnn.X:X, self.cnn.y:y})
            loss += loss_curr
            if (i+1) % print_steps == 0:
                print('Step={} loss={}'.format(i, loss/print_steps))
                loss = 0
                l2_loss = 0
            if (i+1) % save_steps == 0:
                saver.save(sess, './cnn_models/cnn.ckpt', i)


if __name__ == '__main__':
    cnn = TrainCNN(20, 128)
    cnn.train()
