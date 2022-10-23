import pandas
import scipy.io as io
import numpy as np
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
import scipy.stats
from concurrent import futures
import functools
import warnings
warnings.filterwarnings("ignore")

class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")
    
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass

# common_head_NSS_features
# DETCON_exp_1_ckpt_151
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='exp4_IQA_20kvid_512dim_lr_0.1_epoch800_ckpt_250',
                                            help='Evaluated Experiment name.')
    parser.add_argument('--dataset', type=str, default='ShareChat',
                                            help='Evaluation dataset.') 
    parser.add_argument('--use_parallel', action='store_true',
                                            help='Use parallel for iterations.')
    parser.add_argument('--num_iterations', type=int, default=100,
                                            help='Number of iterations of train-test splits')
    parser.add_argument('--max_thread_count', type=int, default=10,
                                            help='Number of threads.')
    args = parser.parse_args()
    return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def compute_metrics(y_pred, y):
    '''
    compute metrics btw predictions & labels
    '''
    # compute SRCC & KRCC
    SRCC = scipy.stats.spearmanr(y, y_pred)[0]
    try:
        KRCC = scipy.stats.kendalltau(y, y_pred)[0]
    except:
        KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

    # logistic regression btw y_pred & y
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)
    
    # compute    PLCC RMSE
    PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
    RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
    return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, params, duration):
    print('======================================================')
    print('params: ', params)
    print('SRCC_train: ', snapshot[0])
    print('KRCC_train: ', snapshot[1])
    print('PLCC_train: ', snapshot[2])
    print('RMSE_train: ', snapshot[3])
    print('======================================================')
    print('SRCC_test: ', snapshot[4])
    print('KRCC_test: ', snapshot[5])
    print('PLCC_test: ', snapshot[6])
    print('RMSE_test: ', snapshot[7])
    print('======================================================')
    print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def final_avg(snapshot):
    def formatted(args, pos):
        mean = np.mean(list(map(lambda x: x[pos], snapshot)))
        stdev = np.std(list(map(lambda x: x[pos], snapshot)))
        print('{}: {} (std: {})'.format(args, mean, stdev))

    print('======================================================')
    print('Average training results among all repeated 80-20 holdouts:')
    formatted("SRCC Train", 0)
    formatted("KRCC Train", 1)
    formatted("PLCC Train", 2)
    formatted("RMSE Train", 3)
    print('======================================================')
    print('Average testing results among all repeated 80-20 holdouts:')
    formatted("SRCC Test", 4)
    formatted("KRCC Test", 5)
    formatted("PLCC Test", 6)
    formatted("RMSE Test", 7)
    print('\n\n')

def evaluate_bvqa_one_split(i, X, y):

    print('{} th repeated holdout test'.format(i))
    t_start = time.time()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=math.ceil(8.8*i))

    X_train_reduced, X_validation, y_train_reduced, y_validation = train_test_split(X_train, y_train, test_size=0.125, random_state=math.ceil(8.8*i))

    alphas = np.logspace(1,2,20)

    SRCC_list = []

    for alpha in alphas :
        regressor = Ridge(alpha = alpha)
        # regressor = LinearRegression()
        
        regressor.fit(X_train_reduced, y_train_reduced)

        y_pred_validation = regressor.predict(X_validation)
        metrics_test = compute_metrics(y_pred_validation , y_validation)

        SRCC_list.append(metrics_test[0])

    idx = np.argmax(np.array(SRCC_list))
    best_alpha = alphas[idx]

    regressor = Ridge(alpha = best_alpha)
    regressor.fit(X_train_reduced,y_train_reduced)
    y_pred_train_reduced = regressor.predict(X_train_reduced)

    y_pred_test = regressor.predict(X_test)

    # compute metrics
    metrics_train = compute_metrics(y_pred_train_reduced, y_train_reduced)
    metrics_test = compute_metrics(y_pred_test, y_test)

    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_alpha, (t_end - t_start))

    return best_alpha, metrics_train, metrics_test
        
def main(args):

    # X = np.load("feature_score_bank/" + args.experiment_name + "/" + args.dataset + "_features.npy")
    # y = np.load("feature_score_bank/" + args.experiment_name + "/" + args.dataset + "_score.npy")
    X_mat = io.loadmat('feature_bank/' + args.dataset + '_features_' + args.experiment_name + '.mat')
    X = np.asarray(X_mat['data'], dtype=np.float64)

    df = pandas.read_csv(args.dataset+'_metadata.csv', skiprows=[], header=None)
    array = df.values
    y = array[1:,1]
    y = y.astype(np.float64)

    # import pdb;pdb.set_trace()
    ## preprocessing
    X[np.isinf(X)] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
    X = imp.transform(X)

    all_iterations = []
    t_overall_start = time.time()
    # 10 times random train-test splits
    if args.use_parallel is True:
        evaluate_bvqa_one_split_partial = functools.partial(
             evaluate_bvqa_one_split, X=X, y=y)
        with futures.ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
            iters_future = [
                    executor.submit(evaluate_bvqa_one_split_partial, i)
                    for i in range(1, args.num_iterations)]
            for future in futures.as_completed(iters_future):
                best_params, metrics_train, metrics_test = future.result()
                all_iterations.append(metrics_train + metrics_test)
    else:
        for i in range(1, args.num_iterations):
            best_params, metrics_train, metrics_test = evaluate_bvqa_one_split(
                    i, X, y)
            all_iterations.append(metrics_train + metrics_test)

    # formatted print overall iterations
    final_avg(all_iterations)
    print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))

if __name__ == '__main__':
    args = arg_parser()
    log_file = "logs3/" + args.experiment_name + "_" + args.dataset + ".txt"
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_file)
    print(args)
    main(args)