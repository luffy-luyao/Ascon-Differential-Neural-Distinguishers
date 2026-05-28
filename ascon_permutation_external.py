import numpy as np
import pandas as pd
from os import urandom
debug = False
debugpermutation = False


def ascon_hash(plaintext, Nr):
    hashlength == 32
    a = b = 3  # rounds
    rate = 8  # bytes
    taglen = 256

    # Initialization
    iv = to_bytes([2, 0, (b << 4) + a]) + int_to_bytes(taglen, 2) + to_bytes([rate, 0, 0])
    S = bytes_to_state(iv + zero_bytes(32))

    S = np.array(S).reshape((1, 5)).astype(dtype=np.uint64)
    t = np.zeros((1, 5), dtype=np.uint64)
    ascon_permutation(S, Nr, t)
    '''
    # Message Processing (Absorbing)
    m_padding = to_bytes([0x01]) + zero_bytes(rate - (len(message) % rate) - 1)
    m_padded = to_bytes(message) + to_bytes(m_padding)
    for block in range(0, len(m_padded), rate):
        A = np.zeros((plaintext.shape[0], 5), dtype=np.uint64)
        t = A
        A = A ^ S
        A[:, 0] ^= bytes_to_int(plaintext[:,0])
        A = ascon_permutation(A, Nr, t)     
    '''
    # message blocks 0,...,n
    A = np.zeros((plaintext.shape[0], 5), dtype=np.uint64)
    t = A
    A = A ^ S
    A[:, 0] ^= plaintext[:, 0]
    A = ascon_permutation(A, Nr, t)

    # Finalization (Squeezing)
    H = np.zeros((plaintext.shape[0], 4), dtype=np.uint64)
    for i in range(4):
        H[:, i] ^= A[:, 0]
        A = ascon_permutation(A, Nr, t)
    return H
def ascon_permutation(S, rounds,t):
    """
    Ascon core permutation for the sponge construction - internal helper function.
    S: Ascon state, a list of 5 64-bit integers
    rounds: number of rounds to perform
    returns nothing, updates S
    """
    assert rounds <= 12
    if debugpermutation: printwords(S, "permutation input:")
    for r in range(12-rounds, 12):
        # --- add round constants ---
        S[:,2] ^= (0xf0 - r *0x10 + r *0x1)
        if debugpermutation: printwords(S, "round constant addition:")
        # --- substitution layer ---
        S[:, 0] ^= S[:, 4]
        S[:, 4] ^= S[:, 3]
        S[:, 2] ^= S[:, 1]
        t[:, 0] = S[:, 0]
        t[:, 1] = S[:, 1]
        t[:, 2] = S[:, 2]
        t[:, 3] = S[:, 3]
        t[:, 4] = S[:, 4]
        t[:, 0] = ~t[:, 0]
        t[:, 1] = ~t[:, 1]
        t[:, 2] = ~t[:, 2]
        t[:, 3] = ~t[:, 3]
        t[:, 4] = ~t[:, 4]
        t[:, 0] &= S[:, 1]
        t[:, 1] &= S[:, 2]
        t[:, 2] &= S[:, 3]
        t[:, 3] &= S[:, 4]
        t[:, 4] &= S[:, 0]
        S[:, 0] ^= t[:, 1]
        S[:, 1] ^= t[:, 2]
        S[:, 2] ^= t[:, 3]
        S[:, 3] ^= t[:, 4]
        S[:, 4] ^= t[:, 0]
        S[:, 1] ^= S[:, 0]
        S[:, 0] ^= S[:, 4]
        S[:, 3] ^= S[:, 2]
        S[:, 2] = ~S[:, 2]
        if debugpermutation: printwords(S, "substitution layer:")
        # --- linear diffusion layer ---
        S[:, 0] ^= rotr(S[:, 0], 19) ^ rotr(S[:, 0], 28)
        S[:, 1] ^= rotr(S[:, 1], 61) ^ rotr(S[:, 1], 39)
        S[:, 2] ^= rotr(S[:, 2], 1) ^ rotr(S[:, 2], 6)
        S[:, 3] ^= rotr(S[:, 3], 10) ^ rotr(S[:, 3], 17)
        S[:, 4] ^= rotr(S[:, 4], 7) ^ rotr(S[:, 4], 41)
        if debugpermutation: printwords(S, "linear diffusion layer:")


    return S


# === helper functions ===

def get_random_bytes(num):
    import os
    return to_bytes(os.urandom(num))

def zero_bytes(n):
    return n * b"\x00"

def ff_bytes(n):
    return n * b"\xFF"

def to_bytes(l): # where l is a list or bytearray or bytes
    return bytes(bytearray(l))

def bytes_to_int(bytes):
    return sum([bi << (i*8) for i, bi in enumerate(to_bytes(bytes))])

def bytes_to_state(bytes):
    return [bytes_to_int(bytes[8*w:8*(w+1)]) for w in range(5)]

def int_to_bytes(integer, nbytes):
    return to_bytes([(integer >> (i * 8)) % 256 for i in range(nbytes)])

def rotr(val, r):
    return (val >> r) | ((val & (1<<r)-1) << (64-r))

def bytes_to_hex(b):
    return b.hex()
    #return "".join(x.encode('hex') for x in b)

def printstate(S, description=""):
    print(" " + description)
    print(" ".join(["{s:016x}".format(s=s) for s in S]))

def printwords(S, description=""):
    print(" " + description)
    print("\n".join(["  x{i}={s:016x}".format(**locals()) for i, s in enumerate(S)]))


def to_binary(S):
    binary_array = np.unpackbits(S.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).reshape(-1, 320)
    return binary_array
def convert_to_binary(c0, c1):
    n = len(c0)
    cb0 = np.zeros((n, 256), dtype=np.uint8)
    cb1 = np.zeros((n, 256), dtype=np.uint8)
    byte_pos = 0
    for pos in range(0, 256, 8):
        byte0 = c0[:, byte_pos].copy()
        byte1 = c1[:, byte_pos].copy()
        for k in range(pos + 7, pos - 1, -1):
            cb0[:, k] = byte0 & 1
            cb1[:, k] = byte1 & 1
            byte0 = byte0 >> 1
            byte1 = byte1 >> 1
        byte_pos += 1
    # cb0_tmp = np.zeros((n, 256), dtype=np.uint8)
    # cb1_tmp = np.zeros((n, 256), dtype=np.uint8)
    # for base in range(0, 256, 64):
    #     for i in range(64):
    #         cb0_tmp[:, base + i] = cb0[:, base + 63 - i]
    #         cb1_tmp[:, base + i] = cb1[:, base + 63 - i]
    return np.concatenate((cb0, cb1), axis=1)
def convert_to_binary(c0, c1):
    n = len(c0)
    cb0 = np.zeros((n, 320), dtype=np.uint8)
    cb1 = np.zeros((n, 320), dtype=np.uint8)
    byte_pos = 0
    for pos in range(0, 320, 8):
        byte0 = c0[:, byte_pos].copy()
        byte1 = c1[:, byte_pos].copy()
        for k in range(pos + 7, pos - 1, -1):
            cb0[:, k] = byte0 & 1
            cb1[:, k] = byte1 & 1
            byte0 = byte0 >> 1
            byte1 = byte1 >> 1
        byte_pos += 1
    # cb0_tmp = np.zeros((n, 256), dtype=np.uint8)
    # cb1_tmp = np.zeros((n, 256), dtype=np.uint8)
    # for base in range(0, 256, 64):
    #     for i in range(64):
    #         cb0_tmp[:, base + i] = cb0[:, base + 63 - i]
    #         cb1_tmp[:, base + i] = cb1[:, base + 63 - i]
    return np.concatenate((cb0, cb1), axis=1)

# === data generation ===

def make_target_diff_samples(num, Nr,df, diff_type):
    p0 = np.frombuffer(urandom(40*num), dtype=np.uint64).copy().reshape(num, 5)
    #p0 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #print(p0)
    diff = np.zeros((num, 5), dtype=np.uint64)
    diff[:,0] = df
    #diff[:,4] = 0x8000000000000000

    #diff = np.zeros(5, dtype=np.uint64)
    #diff[np.arange(0, 5, 5)] = 1
    if diff_type == 1:
        p1 = p0^diff
    else:
        p1 = np.frombuffer(urandom(40*num), dtype=np.uint64).copy().reshape(num, 5)
        #p1 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #p1 = p1.astype(np.uint64)
    T = np.zeros((num, 5), dtype=np.uint64)
    c0 = ascon_permutation(S = p0, rounds=Nr,t=T)
    #T = np.zeros((num, 5), dtype=np.uint64)
    c1 = ascon_permutation(S = p1, rounds=Nr,t=T)
    X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    #print(X.shape)
    return X
def make_target_diff_samples_diff(num, Nr,df, diff_type):
    p0 = np.frombuffer(urandom(40*num), dtype=np.uint64).copy().reshape(num, 5)
    #p0 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #print(p0)
    #diff = np.zeros(5, dtype=np.uint64)
    diff = np.zeros((num,5), dtype=np.uint64)
    diff[:, 0] = df
    #diff[0,:] = 0x0000000000000001
    #diff[4,:] = 0x8000000000000000
    #diff[0,:] = 0x8000100801010000
    #diff[np.arange(0, 5, 5)] = 1
    if diff_type == 1:
        p1 = p0^diff
    else:
        p1 = np.frombuffer(urandom(40*num), dtype=np.uint64).copy().reshape(num, 5)
        #p1 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #p1 = p1.astype(np.uint64)
    T = np.zeros((num, 5), dtype=np.uint64)
    c0 = ascon_permutation(S = p0, rounds=Nr,t=T)
    #T = np.zeros((num, 5), dtype=np.uint64)
    c1 = ascon_permutation(S = p1, rounds=Nr,t=T)
    X = to_binary(c0 ^ c1)
    #X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    #print(X.shape)
    return X

def make_dataset(n, Nr,df):
    num = n // 2
    X_p = make_target_diff_samples(num, Nr,df, diff_type=1)
    X_n = make_target_diff_samples(num, Nr,df, diff_type=0)
    Y_p = [1 for _ in range(num)]
    Y_n = [0 for _ in range(num)]
    X1 = np.concatenate((X_p, X_n), axis=0)
    #print(X1.shape)
    Y0 = np.concatenate((Y_p, Y_n),axis=0)
    Y1 = Y0.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X1,Y1,axis=1)
    np.random.shuffle(X2)
    X = X2[:,:640]
    Y = X2[:,640:]
    #print(Y)
    return X, Y
def make_dataset_diff(n, Nr,df):
    num = n // 2
    X_p = make_target_diff_samples_diff(num, Nr,df, diff_type=1)
    X_n = make_target_diff_samples_diff(num, Nr,df, diff_type=0)
    Y_p = [1 for _ in range(num)]
    Y_n = [0 for _ in range(num)]
    X1 = np.concatenate((X_p, X_n), axis=0)
    #print(X1.shape)
    Y0 = np.concatenate((Y_p, Y_n),axis=0)
    Y1 = Y0.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X1,Y1,axis=1)
    np.random.shuffle(X2)
    X = X2[:,:320]
    #print(X.shape)
    Y = X2[:,320:]
    #print(Y)
    return X, Y
def make_dataset_diff_real(n, Nr,df):
    num = n
    X_p = make_target_diff_samples_diff(num, Nr,df, diff_type=1)
    Y_p = [1 for _ in range(num)]
    Y_p = np.array(Y_p)
    Y1 = Y_p.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X_p,Y1,axis=1)
    np.random.shuffle(X2)
    X = X2[:,:320]
    #print(X.shape)
    Y = X2[:,320:]
    #print(Y)
    return X, Y
def make_dataset_diff_random(n, Nr,df):
    num = n
    X_n = make_target_diff_samples_diff(num, Nr,df, diff_type=0)
    Y_n = [0 for _ in range(num)]
    #print(X1.shape)
    Y_n =np.array(Y_n)
    Y1 = Y_n.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X_n,Y1,axis=1)
    np.random.shuffle(X2)
    X = X2[:,:320]
    #print(X.shape)
    Y = X2[:,320:]
    #print(Y)
    return X, Y
def make_target_mc_diff_samples_diff(num, Nr,s,df, diff_type):
    p0 = np.frombuffer(urandom(40*num*s), dtype=np.uint64).copy().reshape(num*s, 5)
    #p0 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #print(p0)
    #diff = np.zeros(5, dtype=np.uint64)
    diff = np.zeros((num*s,5), dtype=np.uint64)
    diff[:, 0] = df
    #diff[0,:] = 0x0000000000000001
    #diff[4,:] = 0x8000000000000000
    #diff[0,:] = 0x8000100801010000
    #diff[np.arange(0, 5, 5)] = 1
    if diff_type == 1:
        p1 = p0^diff
    else:
        p1 = np.frombuffer(urandom(40*num*s), dtype=np.uint64).copy().reshape(num*s, 5)
        #p1 = np.random.randint(0, 2 ** 64, size=(num, 5), dtype=np.uint64)
    #p1 = p1.astype(np.uint64)
    T = np.zeros((num*s, 5), dtype=np.uint64)
    c0 = ascon_permutation(S = p0, rounds=Nr,t=T)
    #T = np.zeros((num, 5), dtype=np.uint64)
    c1 = ascon_permutation(S = p1, rounds=Nr,t=T)
    X = to_binary(c0 ^ c1)
    #X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    #print(X.shape)
    return X
def make_dataset_mc_diff(n, Nr,s,df):
    num = n // 2
    X_p = make_target_mc_diff_samples_diff(num, Nr,s,df, diff_type=1)
    print(X_p.shape)
    X_p = X_p.reshape(num,320*s)
    print(X_p.shape)
    X_n = make_target_mc_diff_samples_diff(num, Nr,s,df, diff_type=0)
    X_n = X_n.reshape(num,320*s)
    Y_p = [1 for _ in range(num)]
    Y_n = [0 for _ in range(num)]
    X1 = np.concatenate((X_p, X_n), axis=0)
    #print(X1.shape)
    Y0 = np.concatenate((Y_p, Y_n),axis=0)
    Y1 = Y0.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X1,Y1,axis=1)
    np.random.shuffle(X2)
    #X = X2[:,:320*s]
    #print(X.shape)
    #Y = X2[:,320*s:]
    #print(Y)
    return X2
    #return X, Y

'''
def make_td_diff(n, s, nr):
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    t = np.zeros((n*s, 5), dtype=np.uint64)
    state0 = np.random.randint(0, 2**64, size=(n, 5*s), dtype=np.uint64)
    diff = np.zeros(5*s, dtype=np.uint64)
    diff[np.arange(0, 5*s, 5)] = 1
    state1 = state0 ^ diff
    num_rand_samples = np.sum(Y == 0)
    state1[Y == 0, :] = np.random.randint(0, 2**64, size=(num_rand_samples, 5*s), dtype=np.uint64)
    state0 = state0.reshape(n*s, 5)
    state1 = state1.reshape(n*s, 5)
    permutation(state0, nr, t)
    permutation(state1, nr, t)
    X = to_binary(state0 ^ state1)
    return X, Y


def make_real_data(s):
    t = np.zeros((s, 5), dtype=np.uint64)
    state0 = np.random.randint(0, 2 ** 64, size=(s, 5), dtype=np.uint64)
    diff = np.zeros((s, 5), dtype=np.uint64)
    diff[:, 0] = 1
    state1 = state0 ^ diff
    permutation(state0, 4, t)
    permutation(state1, 4, t)
    X = to_binary(state0 ^ state1)
    return X
'''
def save_dataset_csv(X, Y, filename='simulated_dataset.csv'):
    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = Y.flatten()  
    df.to_csv(filename, index=False)
    print(f"save dataset to {filename}")



if __name__ == '__main__':
    for i in range(4,5,1):
        X, Y = make_dataset(n=10 ** 6, Nr=i, df=0x0000000000000001)
        save_dataset_csv(X, Y, 'ascon_simulated_data.csv')


