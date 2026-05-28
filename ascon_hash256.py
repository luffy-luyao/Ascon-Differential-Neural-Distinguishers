from __future__ import annotations
import numpy as np
import pandas as pd
from os import urandom


debug = False
debugpermutation = False






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
        '''
        if r%2==1:
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
        else:
            # --- linear diffusion layer ---
            S[:, 0] ^= rotr(S[:, 0], 19) ^ rotr(S[:, 0], 28)
            S[:, 1] ^= rotr(S[:, 1], 61) ^ rotr(S[:, 1], 39)
            S[:, 2] ^= rotr(S[:, 2], 1) ^ rotr(S[:, 2], 6)
            S[:, 3] ^= rotr(S[:, 3], 10) ^ rotr(S[:, 3], 17)
            S[:, 4] ^= rotr(S[:, 4], 7) ^ rotr(S[:, 4], 41)
            if debugpermutation: printwords(S, "linear diffusion layer:")
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
            '''

    return S
# === Ascon hash/xof ===

# plaintext.shape = (n, N * r) = (n, N * 136)
# return: ciphertext.shape = (n, 32)
def ascon_hash(plaintext, Nr):


    hashlength = 32
    a = b = Nr  # rounds
    rate = 8  # bytes
    taglen = 256

    # Initialization
    #iv = to_bytes([2, 0, (b << 4) + a]) + int_to_bytes(taglen, 2) + to_bytes([rate, 0, 0])
    iv = int_to_bytes(0x0000080100cc0002,8)
    S = bytes_to_state(iv + zero_bytes(32))

    S = np.array(S).reshape((1,5)).astype(dtype=np.uint64)
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
    A[:, 0] ^= plaintext[:,0]
    A = ascon_permutation(A, Nr, t)

    # Finalization (Squeezing)
    H = np.zeros((plaintext.shape[0],4), dtype=np.uint64)
    for i in range(4):
        H[:,i] ^=A[:,0]
        A = ascon_permutation(A, Nr,t)
    return H
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



def to_binary(A):
    binary_array = np.unpackbits(A.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).reshape(-1, 256)
    return binary_array

def convert_to_binary(c0, c1):
    n = len(c0)
    cb0 = np.zeros((n, 256), dtype=np.uint8)
    cb1 = np.zeros((n, 256), dtype=np.uint8)
    byte_pos = 0
    for pos in range(0, 256, 8):
        byte0 = c0[:,byte_pos].copy()
        byte1 = c1[:,byte_pos].copy()
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

def make_target_diff_samples(num, Nr, diff_type):
    #p0_copy = np.frombuffer(urandom(8*n), dtype=np.uint64).reshape(n, 1)
    #p0 = np.copy(p0_copy)
    p0 = np.frombuffer(urandom(8 * num), dtype=np.uint64).copy().reshape(num, 1)
    #print(p0)
    diff = np.zeros((num, 1), dtype=np.uint64)
    diff[:,0] = 0x8000000000000000
    if diff_type == 1:
        p1 = p0^diff
        #print(p1)
    else:
        p1 = np.frombuffer(urandom(8 * num), dtype=np.uint64).copy().reshape(num, 1)
    #print(p0.shape)
    c0 = ascon_hash(p0, Nr)
    c1 = ascon_hash(p1, Nr)
    X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    #X = convert_to_binary(c0, c1)
    return X
def make_target_diff_padd_samples(num, Nr, diff_type):
    #p0_copy = np.frombuffer(urandom(8*n), dtype=np.uint64).reshape(n, 1)
    #p0 = np.copy(p0_copy)
    p0 = np.frombuffer(urandom(8 * num), dtype=np.uint8).copy().reshape(num, 8)
    p0[:,0]=128
    p0 = p0.copy().view(np.uint64).reshape(num,1)
    #print(p0)
    diff = np.zeros((num, 1), dtype=np.uint64)
    diff[:,0] = 0x8000000000000000
    if diff_type == 1:
        p1 = p0^diff
        #print(p1)
    else:
        p1 = np.frombuffer(urandom(8 * num), dtype=np.uint64).copy().reshape(num, 1)
        p1[:, 0] = 128
        p1 = p0.copy().view(np.uint64).reshape(num, 1)
    #print(p0.shape)
    c0 = ascon_hash(p0, Nr)
    c1 = ascon_hash(p1, Nr)
    X = np.concatenate((to_binary(c0), to_binary(c1)), axis=1)
    #X = convert_to_binary(c0, c1)
    return X
def make_dataset(n, Nr):
    num = n // 2
    #X_p = make_target_diff_padd_samples(num, Nr, 1)
    #X_n = make_target_diff_padd_samples(num, Nr, 0)
    X_p = make_target_diff_samples(num, Nr, 1)
    X_n = make_target_diff_samples(num, Nr, 0)
    Y_p = [1 for _ in range(num)]
    Y_n = [0 for _ in range(num)]
    X1 = np.concatenate((X_p, X_n), axis=0)
    #print(X1.shape)
    Y0 = np.concatenate((Y_p, Y_n),axis=0)
    Y1 = Y0.reshape(-1,1)
    #print(Y1.shape)
    X2 = np.append(X1,Y1,axis=1)
    np.random.shuffle(X2)
    X = X2[:,:512]
    Y = X2[:,512:]
    #print(Y)
    return X, Y

# === some demo if called directly ===

def demo_print(data):
    maxlen = max([len(text) for (text, val) in data])
    for text, val in data:
        print("{text}:{align} 0x{val} ({length} bytes)".format(text=text, align=((maxlen - len(text)) * " "), val=bytes_to_hex(val), length=len(val)))


def demo_hash(variant="Ascon-Hash256", hashlength=32):
    assert variant in ["Ascon-Hash256", "Ascon-XOF128", "Ascon-CXOF128"]
    print("=== demo hash using {variant} ===".format(variant=variant))

    message = b"ascon"
    tag = ascon_hash(message,Nr=12)
    demo_print([("message", message), ("tag", tag)])



def save_dataset_csv(X, Y, filename='simulated_dataset.csv'):
 

    feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)

    df['label'] = Y.flatten()  
    
    df.to_csv(filename, index=False)
    print(f"save dataset to {filename}")



if __name__ == '__main__':
    for i in range(4,5,1):
        X, Y = make_dataset(n=10 ** 6, Nr=i)
        save_dataset_csv(X, Y, 'ascon_simulated_data.csv')




