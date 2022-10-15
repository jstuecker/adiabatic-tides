import h5py
import numpy as np

def h5py_write_dict(file, datadict, path="", overwrite=False, verbose=True):
    """ This function can write recursive dicts to an hdf5 file
    
    file : either a filename or an already open hdf5 file
    datadict : a dictionary to save to the file
    path : path inside the file to save the dictionary to. E.g.
           "/path/to/mydictionary"
    overwrite : whether to allow overwriting existing datasets.
    verbose : degree of verbosity (can be True or False)
    """
    if type(file) is str: # Filename was provided -> open as hdf5 file
        with h5py.File(file, "a") as myfile:
            return h5py_write_dict(myfile, datadict, path=path, overwrite=overwrite, verbose=verbose)
    
    if path != "":
        if not path in file:
            file.create_group(path)
        file[path].attrs["type"] = "dict"

        file = file[path]

    for dictkey in datadict.keys():
        if type(dictkey) == type(1):
            key = "_int_%d" % dictkey
        elif type(dictkey) == type(1.2):
            key = "_float_%g" % dictkey
        else:
            key = dictkey

        if type(datadict[dictkey]) is dict:
            file.require_group(key)
            h5py_write_dict(file[key], datadict[dictkey], overwrite=overwrite, verbose=verbose)
        else:
            if (key in file) and not overwrite:
                if verbose > 0:
                    print("Ignoring existing %s (overwrite=False) " % (file[key].name, ))
            else:
                if key in file:
                    if verbose > 0:
                        print("Overwriting %s " % (file[key].name,))
                    del file[key]
                file.create_dataset(key, data=datadict[dictkey])
                
def h5py_read_dict(file, path="", datadict=None, verbose=True):
    """ This function can read dictionaries that were written to an hdf5 file by h5py_write_dict
    
    path : path inside the file to save the dictionary to. E.g.
           "/path/to/mydictionary"
    datadict : Can be a pre-existing dictionry that the results should be inserted to.
               If None : a new dictionary instance will be created.
    verbose : degree of verbosity (can be True or False)
    
    returns : the read dictionary
    """
    if datadict is None:
        datadict = {}

    if type(file) is str: # Filename was provided -> open as hdf5 file
        with h5py.File(file, "r") as myfile:
            return h5py_read_dict(myfile, path=path, datadict=datadict, verbose=verbose)
        
    if path != "":
        file = file[path]

    for key in file.keys():
        if key[0:5] == "_int_":
            dictkey = int(key[5:])
        elif key[0:7] == "_float_":
            dictkey = float(key[7:])
        else:
            dictkey = key

        if type(file[key]) is h5py._hl.group.Group:
            datadict[dictkey] = {}
            h5py_read_dict(file[key], datadict=datadict[dictkey], verbose=verbose)
        elif type(file[key]) is h5py._hl.dataset.Dataset:
            datadict[dictkey] = np.array(file[key])
        else:
            print("Ignoring unhandled type of %s :" % file[key].name, type(file[key]))

    return datadict

def h5py_print_file_structure(file):
    """A Convenience function that prints the file-structure of an existing hdf5 file
    
    file : A filename of an hdf5 file or opened hdf5 file"""
    if type(file) is str: # Filename was provided -> open as hdf5 file
        with h5py.File(file, "r") as myfile:
            return h5py_print_file_structure(myfile)

    for key in file.keys():
        mytype = h5py_get_type(file, key)
        if mytype == "group":
            h5py_print_file_structure(file[key])
        else:
            print("%s (%s)" % (file[key].name, mytype))
            
def h5py_get_type(file, path):
    """Returns the type of object sitting on some path of an hdf5 file
    
    file : already open hdf5 file
    path : path to check
    
    returns : either "data" or "group"
    """
    if type(file[path]) == h5py._hl.dataset.Dataset:
        return "data"
    elif type(file[path]) == h5py._hl.group.Group:
        if "type" in file[path].attrs:
            return file[path].attrs["type"]
        else:
            return "group"
    else:
        assert 0, "Unknown Type"