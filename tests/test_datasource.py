import sys
sys.path.append('..')
import datasource
import pylab as pl


def test_narrow_spectra():
    spectra = datasource.SpectrumTableNarrow('../FiGeLMC_R100_sn5.csv')
    spectra._shuffle_data()
    spectra._split_data()
    print spectra.target
    for i in [0, 100, 300, 600, 800, -1]:
        pl.plot(spectra.data[i, :], label='%.1f' % spectra.target[i])
    pl.legend()
    pl.show()


def test_wide_spectra():
    spectra = datasource.SpectrumTableWide('../FiGeLMC_R100_sn5_wide.csv',
                                           target_name='fesc', exclude_cols=['m_ab'])
    print spectra.num_objs
    print spectra.num_features
    print spectra.data.shape
    print spectra.target
    for i in [0, 100, 300, 600, 800, -1]:
        pl.plot(spectra.data[i, :], label='%.1f' % spectra.target[i])
    pl.legend()
    pl.show()


if __name__ == '__main__':
    test_wide_spectra()