from get_stock_data import download_all

#Download Stocks
download_all()



import os
import pandas as pd
from gan import GAN
import random
import tensorflow as tf
import xgboost as xgb
from sklearn.externals import joblib


os.environ["CUDA_VISIBLE_DEVICES"]=""

class Predict:

    def __init__(self, num_historical_days=20, days=10, pct_change=0, gan_model='./deployed_models/gan', cnn_modle='./deployed_models/cnn', xgb_model='./deployed_models/xgb'):
        self.data = []
        self.num_historical_days = num_historical_days
        self.gan_model = gan_model
        self.cnn_modle = cnn_modle
        self.xgb_model = xgb_model
        # assert os.path.exists(gan_model)
        # assert os.path.exists(cnn_modle)
        # assert os.path.exists(xgb_model)

        files = [os.path.join('./stock_data', f) for f in os.listdir('./stock_data')]
        for file in files:
            print(file)
            df = pd.read_csv(file, index_col='Date', parse_dates=True)
            df = df[['Open','High','Low','Close','Volume']]
            df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
            df = df.dropna()
            self.data.append((file.split('/')[-1], df.index[0], df[200:200+num_historical_days].values))


    def gan_predict(self):
    	tf.reset_default_graph()
        gan = GAN(num_features=5, num_historical_days=self.num_historical_days,
                        generator_input_size=200, is_train=False)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, self.gan_model)
            clf = joblib.load(self.xgb_model)
            for sym, date, data in self.data:
	            features = sess.run(gan.features, feed_dict={gan.X:[data]})
	            features = xgb.DMatrix(features)
	            print('{} {} {}'.format(str(date).split(' ')[0], sym, clf.predict(features)[0][1] > 0.5))
	            


if __name__ == '__main__':
	p = Predict()
	p.gan_predict()
