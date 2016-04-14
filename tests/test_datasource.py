import sys
sys.path.append('..')
import datasource


def test_load_file():
    data_source = datasource.DataSource('../FiGeLMC_R100_sn5.csv')


if __name__ == '__main__':
    test_load_file()