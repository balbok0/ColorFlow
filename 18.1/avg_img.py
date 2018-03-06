from methods import *
from get_fnames import get_raw_names
import time


# Show an average image of array from file fname.
# Shows an image with output_name title, and saves it as output_name.
def avg_img(gen):
    fname = get_raw_names()[gen]
    ts = time.time()
    data0 = HDF5Matrix(fname[0], 'images')
    np.save('images/avg_img/' + gen + " 0", mean(data0))
    print "Time it took for 0\'s of " + gen + " " + "{:.3}".format(time.time()-ts) + "s"

    ts = time.time()
    data1 = HDF5Matrix(fname[1], 'images')
    np.save('images/avg_img/' + gen + " 1", mean(data1))
    print 'Time it took for 1\'s of ' + gen + " " + "{:.3}".format(time.time()-ts) + "s"
    # plt.xlabel("Prop. to pseudorapidity")
    # plt.ylabel("Prop. to translated azimuthal angle")
    # plt.title(output_name)
    # plt.colorbar(fig)
    # plt.savefig("images/average " + output_name)
    # plt.show()
    # plt.close()


def mean(data):
    split_n = int(np.ceil(len(data) / 1000.0))
    sum = []
    for i in range(25):
        sum.append([])
        for j in range(25):
            sum[i].append(0.0)
    sum = np.array(sum, dtype=np.float64)
    for i in range(split_n):
        temp = data[i*1000:i*1000+1000]
        sum = np.add(sum, temp)
    return np.divide(sum, data.shape[0])


#avg_img("data/Sherpa/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Sherpa JZ")
#avg_img("data/Sherpa/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Sherpa WZ")

#avg_img("data/Pythia/Vincia/qcd_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Pythia Vincia QCD")
#avg_img("data/Pythia/Vincia/w_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Pythia Vincia W")

#avg_img("data/Herwig/Angular/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Herwig Angular JZ")
#avg_img("data/Herwig/Angular/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5", "Herwig Angular WZ")

#avg_img("data/Herwig/Dipole/QCD_Dipole250-300_j1p0_sj0p30_delphes_jets_images.h5", "Herwig Dipole QCD")
#avg_img("data/Herwig/Dipole/WZ_combined_j1p0_sj0p30_delphes_jets_images.h5", "Herwig Dipole WZ")

for gen in generators:
    avg_img(gen)
