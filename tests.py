import lasso_tf

if __name__ == '__main__':
    dataset, labels = lasso_tf.get_test_data()
    print dataset.shape
    print labels.shape