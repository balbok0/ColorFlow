import os

# This is the only line which has to be changed, according to where data is kept.
drive_path = '../Data/'

generators = ['Herwig Angular', 'Herwig Dipole', 'Sherpa',
              'Pythia Standard', 'Pythia Vincia']


# Checks if drive is plugged in.
def check_drive():
    return os.path.exists(drive_path)


# If drive is plugged in returns raw data, from this drive.
# If not, raises IOError.
def get_raw_names():
    files = {}
    raw_dir = drive_path + 'raw data/'
    if check_drive():
        files['Pythia Standard'] = [raw_dir + "Pythia/Standard/qcd_j1p0_sj0p30_delphes_jets_pileup_images.h5",
                                    raw_dir + "Pythia/Standard/w_j1p0_sj0p30_delphes_jets_pileup_images.h5"]
        files['Pythia Vincia'] = [raw_dir + "Pythia/Vincia/qcd_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5",
                                  raw_dir + "Pythia/Vincia/w_vincia_j1p0_sj0p30_delphes_jets_pileup_images.h5"]
        files['Herwig Angular'] = [raw_dir + "Herwig/Angular/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5",
                                   raw_dir + "Herwig/Angular/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5"]
        files['Herwig Dipole'] = [raw_dir + "Herwig/Dipole/QCD_Dipole250-300_j1p0_sj0p30_delphes_jets_images.h5",
                                  raw_dir + "Herwig/Dipole/WZ_combined_j1p0_sj0p30_delphes_jets_images.h5"]
        files['Sherpa'] = [raw_dir + 'Sherpa/JZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5',
                           raw_dir + 'Sherpa/WZ_combined_j1p0_sj0p30_delphes_jets_pileup_images.h5']
    else:
        raise IOError("Drive is not inserted.")
    return files


# If drive is plugged in returns pre-processed data, from this drive.
# If not, raises IOError.
def get_ready_names():
    files = {}
    if check_drive():
        for gen in generators:
            files[gen] = drive_path + "reco data/" + gen.replace(' ', '/') + '/data.h5'
    else:
        raise IOError("Drive is not inserted.")
    return files


# Returns paths to toy datasets
def get_toy_names():
    files = {}
    for gen in generators:
        gen_path = 'toy/' + gen.replace(' ', '/') + '/data.h5'
        if not os.path.exists(gen_path):
            raise IOError("Generator " + gen + " not found, at path: " + gen_path)
        files[gen] = gen_path
    return files


# Returns dictionary of model type for all generators.
# If they do not exist, raises IOError.
def get_model_names(model_name):
    models = {}
    models_path = 'models/'
    for gen in generators:
        if os.path.exists("{path}validated {mod} {g}".format(path=models_path, mod=model_name, g=gen)):
            models[gen] = ("{path}validated {mod} {g}".format(path=models_path, mod=model_name, g=gen))
        else:
            raise IOError(gen + " model not found. Should be in form: models/validated " +
                          model_name + " " + gen)
    return models


# Returns dictionary of colors, for each generator.
# Colors are meant to be like ones in paper.
def get_colors():
    return dict(zip(generators, ['r-.', 'm--', 'gold', 'b', 'g--']))


# Returns list of generators.
def get_generators():
    return generators
