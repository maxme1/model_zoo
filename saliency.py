from tqdm import tqdm
import numpy as np

def generate_pathces(x, shape, count_per_entry):
#     per_entry = (count - 1) // len(x) + 1
    for y in x:
        upper = np.array(y.shape) - shape + 1
        for _ in range(count_per_entry):
#             if count <= 0:
#                 return
            begin = [np.random.randint(0, i) for i in upper]
            end = begin + np.array(shape)
            idx = [slice(*i) for i in zip(begin, end)]
            yield y[idx]
#             count -= 1

def sampler(x, shape, inner_shape, count):
    begin = (np.array(shape) - inner_shape) // 2
    end = begin + inner_shape
    coords = [slice(*i) for i in zip(begin, end)]
    
    A = b = n = 0
    negation = None
    for y in generate_pathces(x, shape, count):
        p1 = y[coords]

        if negation is None:
            negation = np.ones(y.shape, dtype=bool)
            negation[coords] = False

        p2 = y[negation]
        y = np.hstack((p1.reshape(-1), p2.reshape(-1)))

        A += np.tensordot(y, y, axes=0)
        b += y
        n += 1

    b /= n
    A = A / (n - 1) - n / (n - 1) * np.tensordot(b, b, axes=0)
    
    size = np.prod(inner_shape)
    S_11 = A[:size, :size]
    S_12 = A[:size, size:]
    S_22 = A[size:, size:]
    prod = S_12.dot(np.linalg.inv(S_22))
    mu = b[:size] - prod.dot(b[size:])
    S = S_11 - prod.dot(S_12.T)
    
    def wrapped(a, size):
        a = a[negation].reshape(-1)
        ret = np.random.multivariate_normal(mu + prod.dot(a), S, size)
        return ret.reshape((size,) + inner_shape)
    return wrapped

def odds(x):
    return x / (1 - x)

def get_saliency(network, source, outer_shape, inner_shape, batch_size, sample, num_samples, step=1, verbose=False):
    begin = (np.array(outer_shape) - inner_shape) // 2
    end = begin + inner_shape
    coords = [slice(*i) for i in zip(begin, end)]
    
    dims = np.array(source.shape) - outer_shape + 1
    dims = dims // step
    dims[dims == 0] = 1
    
    probs = np.zeros(dims)
    real = network([source])[0]

    batch = []
    indexes = []
    
    # calculate the probabilities
    bar = np.ndindex(*dims)
    if verbose:
        bar = tqdm(bar)
    for corner in bar:
        real_corner = np.array(corner) * step
        start = real_corner + begin
        idx = [slice(*i) for i in zip(start, start + inner_shape)]
        surround = [slice(*i) for i in zip(real_corner, real_corner + outer_shape)]
        surround = source[surround]

        for s in sample(surround, num_samples):
            temp = source.copy()
            temp[idx] = s
            batch.append(temp)
            indexes.append(corner)
            if len(batch) == batch_size:
                prob = network(batch)
                for key, value in zip(indexes, prob):
                    probs[key] += value
                batch = []
                indexes = []
                
    if batch:
        prob = network(batch)
        for key, value in zip(indexes, prob):
            probs[key] += value
    probs /= num_samples
     
    # create the attention map
    probs = np.log(odds(probs))
    real = np.log(odds(real))
    saliency = np.zeros_like(source)
    counts = np.zeros_like(source)
    
    for corner in np.ndindex(*dims):
        real_corner = np.array(corner) * step
        start = real_corner + begin
        idx = [slice(*i) for i in zip(start, start + inner_shape)]

        saliency[idx] += real - probs[corner]
        counts[idx] += 1
        
    counts[counts == 0] = 1    
    return saliency / counts / np.log(2)