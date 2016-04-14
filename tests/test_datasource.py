import sys
sys.path.append('..')
import datasource
import pylab as pl


def test_narrow_spectra():
    spectra = datasource.NarrowSpectraTable('../FiGeLMC_R100_sn5.csv')
    spectra.shuffle_data()
    spectra.split_data()
    print spectra.target
    for i in [0, 100, 300, 600, 800, -1]:
        pl.plot(spectra.data[i, :], label='%.1f' % spectra.target[i])
    pl.legend()
    pl.show()


if __name__ == '__main__':
    test_narrow_spectra()