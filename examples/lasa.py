import numpy as np
import scipy.io
import os


def load_demos(shape_idx):
    """Load demonstrations from LASA dataset.

    The LASA dataset is available at
    http://lasa.epfl.ch/khansari/LASA_Handwriting_Dataset.zip
    it is explained at
    http://lasa.epfl.ch/people/member.php?SCIPER=183746/#SEDS_Benchmark_Dataset

    Parameters
    ----------
    shape_idx : int
        Choose demonstrated shape, must be within range(30).

    Returns
    -------
    X : array-like, shape (n_task_dims, n_steps, n_demos)
        Positions

    Xd : array-like, shape (n_task_dims, n_steps, n_demos)
        Velocities

    Xdd : array-like, shape (n_task_dims, n_steps, n_demos)
        Accelerations

    dt : float
        Time between steps

    shape_name : string
        Name of the Matlab file from which we load the demonstrations
        (without suffix)
    """
    dataset_path = os.sep.join(__file__.split(os.sep)[:-1])
    if dataset_path != "":
        dataset_path += os.sep
    dataset_path += "lasa" + os.sep + "DataSet" + os.sep
    demos, shape_name = load_from_matlab_file(dataset_path, shape_idx)
    X, Xd, Xdd, dt = convert_demonstrations(demos)
    return X, Xd, Xdd, dt, shape_name


def load_from_matlab_file(dataset_path, shape_idx):
    """Load demonstrations from Matlab files."""
    file_name = sorted(os.listdir(dataset_path))[shape_idx]
    return (scipy.io.loadmat(dataset_path + file_name)["demos"][0],
            file_name[:-4])


def convert_demonstrations(demos):
    """Convert Matlab struct to numpy arrays."""
    tmp = []
    for demo_idx in range(demos.shape[0]):
        # The Matlab format is strange...
        demo = demos[demo_idx][0, 0]
        # Positions, velocities and accelerations
        tmp.append((demo[0], demo[2], demo[3]))

    X = np.transpose([P for P, _, _ in tmp], [1, 2, 0])
    Xd = np.transpose([V for _, V, _ in tmp], [1, 2, 0])
    Xdd = np.transpose([A for _, _, A in tmp], [1, 2, 0])

    dt = float(demos[0][0, 0][4])

    return X, Xd, Xdd, dt
