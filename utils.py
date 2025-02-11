import pickle


def save_list(l, path):
    if not path.endswith('.pkl'):
        path = path + '.pkl'
        print('Given name not ended with .pkl. Add .pkl as postfix.')

    with open(path, 'wb') as f:
        pickle.dump(l, f)

def load_list(path):
    if not path.endswith('.pkl'):
        raise NameError('Given name not ended with .pkl. CANNOT load.')

    with open(path, 'rb') as f:
        l = pickle.load(f)
        return l
