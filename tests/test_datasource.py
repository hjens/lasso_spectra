import sys
sys.path.append('..')
import datasource
import pylab as pl

def test_load_file():
    #data_source = datasource.DataSource('../FiGeLMC_R100_sn5.csv')
    #print data_source.colnames
    #print data_source.data['gal_id']
    pass

def test_narrow_spectra():
    spectra = datasource.NarrowSpectraTable('../FiGeLMC_R100_sn5.csv')
    for i in [0, 100, 300, 600, 800, -1]:
        pl.plot(spectra.design_matrix[i,:])
    pl.show()

if __name__ == '__main__':
    test_load_file()
    test_narrow_spectra()