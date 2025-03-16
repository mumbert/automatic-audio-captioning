import pickle

def save_pickle(filename, data):

    print(f"Saving data in pickle file: {filename}")

    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def load_pickle(filename):

    print(f"Loading data from pickle file: {filename}")

    with open (filename, 'rb') as fp:
        data = pickle.load(fp)

    return data