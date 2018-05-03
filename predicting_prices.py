import os
import time
import pandas as pd
import logging
import math

from statsmodels.tsa.ar_model import AR


DATA_PATH = os.path.join(os.getcwd(), 'data', 'productsPrices.csv')


# This is a time series regression problem where I need to forecast the next day price

def _logger(file_name='predicting_prices.log'):
    level = logging.INFO
    format = '%(asctime)s - %(process)s-%(thread)-6.6s - %(filename)-15.15s %(funcName)-15.15s ' \
             '%(lineno)-4.4s - %(levelname)-6.6s - %(message)s'
    handlers = [logging.FileHandler(file_name), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    return logging


class Timer(object):
    """
    This class returns timer values.
    """
    def __init__(self, p_hours=True, p_minutes=True, p_seconds=True):
        self.end = None
        self.start = None
        self.start_stop_timer()
        self.p_hours = p_hours
        self.p_minutes = p_minutes
        self.p_seconds = p_seconds

    def print_timer(self):
        self.start_stop_timer(start=False)
        hours, rem = divmod(self.end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.p_seconds:
            if self.p_minutes:
                if self.p_hours:
                    return "%d:%02d:%02d (hh:mm:ss)" % (hours, minutes, seconds)
                else:
                    return "%02d:%02d (mm:ss)" % (minutes, seconds)
            else:
                return "%02d (ss)" % seconds

        if self.p_minutes:
            if self.p_hours:
                return "%d:%02d (hh:mm)" % (hours, minutes)
            else:
                return "%02d (mm)" % minutes

        if self.p_hours:
            return "%2d (hh)" % hours

    def start_stop_timer(self, start=True):
        if start:
            self.start = time.time()
        else:
            self.end = time.time()


class PredictPrices(object):
    """
    This class trains on a time series data and forecast the next day value
    """
    def __init__(self, data_path=None, inspect_data=False, n_estimators=10, number_of_test_values=1):
        """
        :param data_path:string The location of the data file (loads CSV files only)
        :param inspect_data:bool Go over all the input values and make sure everything comes as expected
        :param n_estimators:int The amount of estimators to use
        :param number_of_test_values:int The amount of given test values
        """
        self.logger.info('Prices prediction process begins')
        self.logger = _logger()
        self.data_path = data_path
        self.number_of_test_values = number_of_test_values
        self.inspect_data = inspect_data
        self.n_estimators = n_estimators
        self.validate_input()
        self.products = []
        self.model = None
        self.model_fit = None
        self.t = Timer()

    def train(self, x_train, x_test):
        self.model = AR(x_train)
        history = [x_train[i] for i in range(len(x_train))]
        min_diff = math.inf
        optimized_maxlag = 0
        best_trend = vanilla_predictor = None
        for i in range(1, len(x_train)):
            for trend in [None, 'nc']:
                for vanilla in [True, False]:
                    if trend is None:
                        self.model_fit = self.model.fit(maxlag=i, disp=False)
                    else:
                        self.model_fit = self.model.fit(maxlag=i, disp=False, trend=trend)
                    y_predicted = self.predict(history, vanilla_predictor=vanilla)
                    temp_diff = abs(y_predicted - x_test)

                    if temp_diff < min_diff:
                        best_trend = trend
                        min_diff = temp_diff
                        optimized_maxlag = i
                        vanilla_predictor = vanilla
        if best_trend is None:
            self.model_fit = self.model.fit(maxlag=optimized_maxlag, disp=False)
        else:
            self.model_fit = self.model.fit(maxlag=optimized_maxlag, disp=False, trend=best_trend)
        return self.model_fit, history, vanilla_predictor

    def predict(self, history, vanilla_predictor=False):
        coef = self.model_fit.params
        if vanilla_predictor:
            return self.model.predict(params=coef)[0]

        yhat = coef[0]
        for i in range(1, len(coef)):
            yhat += coef[i] * history[-i]
        return yhat

    def import_and_arrange_data(self):
        train = pd.read_csv(self.data_path)
        train.fillna(
            train.mean())  # Empty cells (if there are any) will be replaced with the average of the product prices
        if self.inspect_data:
            self.logger.info(train.describe())
        self.products = train.columns.tolist()

        values_per_product = {}
        for product in self.products:
            values_per_product[product] = train[product].values.tolist()
        return values_per_product

    def import_train_and_predict(self):
        t = Timer()
        x_test_predicted = []
        x_test = []
        train_per_product = self.import_and_arrange_data()

        for product, values in train_per_product.items():
            if self.number_of_test_values >= len(values):
                raise Exception('The number of test values must be smaller than the amount of train set values')

            x_train = values[:-self.number_of_test_values]
            x_test.append(values[-self.number_of_test_values])
            self.model_fit, history, vanilla_predictor = self.train(x_train, values[-self.number_of_test_values])
            x_test_predicted.append(self.predict(history, vanilla_predictor))

        for x_test_, x_predicted in zip(x_test, x_test_predicted):
            self.logger.info('Predicted value: %.3f, real value: %s' % (x_predicted, x_test_))
            self.logger.info('Abs diff between predicted and true value: %.3f\n' % (abs(x_test_ - x_predicted)))

        self.logger.info('Price prediction took %s' % t.print_timer())

    def validate_input(self):
        self.logger.info('Validating input')
        if self.data_path is None:
            raise Exception('You must provide the location of the data')
        if self.data_path[-3:] != 'csv':
            raise TypeError('This code support CSV files only')
        if self.number_of_test_values <= 0:
            raise ValueError('The number of test values must be greater than 0')
        if not isinstance(self.number_of_test_values, int):
            raise TypeError('The number of test values must be integer')
        if not isinstance(self.inspect_data, bool):
            raise TypeError('Inspect data must be Boolean')
        if not isinstance(self.n_estimators, int):
            raise TypeError('The number of estimators must an integer')
        self.logger.info('Validation complete')


if __name__ == '__main__':
    pp = PredictPrices(data_path=DATA_PATH)
    pp.import_train_and_predict()
