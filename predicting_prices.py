import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.getcwd(), 'data', 'productsPrices.csv')


class PredictPrices(object):
    def __init__(self, test_size=.2, data_path=None, inspect_data=False, visualize_data=False):
        self.logger = logging.getLogger('predict_prices_logger')
        self.logger.info('Prices prediction process starts')
        self.data_path = data_path
        self.test_size = test_size
        self.inspect_data = inspect_data
        self.visualize_data_ = visualize_data  # Create histogram (https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f)
        self.validate_input()
        self.products = []

    def train(self, x_train, y_train):
        pass

    def predict(self, x_predict):
        pass

    def import_and_arrange_data(self):
        train = pd.read_csv(self.data_path)
        train.fillna(train.mean())  # Empty cells will be replaced with the average of the product prices
        if self.inspect_data:
            self.logger.info(train.describe())
        self.products = train.columns.tolist()

        values_per_product = {}
        for product in self.products:
            values_per_product[product] = train[product].values.tolist()
        return values_per_product

    def import_train_and_predict(self):
        train_per_product = self.import_and_arrange_data()
        for product, values in train_per_product.items():
            x_train, x_test, y_train, y_test = train_test_split(values, test_size=self.test_size, random_state=2)

        pass

    def visualize_data(self):
        pass

    def validate_input(self):
        if self.data_path is None:
            raise Exception('You must provide the location of the data')
        if not isinstance(self.test_size, float):
            raise TypeError('Test size must be given as a float')
        if self.test_size < 0 or self.test_size > 1.:
            raise ValueError('Test size must be between 0.0 - 1.0')
        if not isinstance(self.inspect_data, bool):
            raise TypeError('Inspect data must be Boolean')
        if not isinstance(self.visualize_data_, bool):
            raise TypeError('Visualize data must be Boolean')



# Product 1:13019
# Product 2: 26266
# Product 3: 40993
# Product 4: 3005058
# Product 5: 22685
# Product 6: 33372
# Product 7: 25496
# Product 8: 145638
# Product 9: 166472
# Product 10: 201760



if __name__ == '__main__':
    pp = PredictPrices(test_size=.2, data_path=DATA_PATH)
    pp.train_predict_and_print()