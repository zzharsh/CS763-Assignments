import argparse
import numpy as np

# 1. Create specified matrix of dimension N:
def ccreate(N):
    a = np.zeros((N,N))
    i = np.eye(N)
    j1 = np.arange(0,N,2)
    j2 = np.arange(1,N,2)
    if N%2==0:
      N = int(N/2)
    else:
      N = int(N/2)+1
    a[0:N] = i[j1]
    a[N:] = i[j2]

    return a

#2.
def crop_array(arr_2d, offset_height, offset_width, target_height, target_width):
    result = arr_2d[offset_height: offset_height+target_height, offset_width: offset_width+target_width]
    return result


#3.
def padd(arr_2d):
    result = np.pad(arr_2d,2, 'constant', constant_values=0.5)
    return result

#4.
def cconcatenate(arr_2d):
    result = np.concatenate((arr_2d, arr_2d), axis= 1)
    return result

#5.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int)
    args = parser.parse_args()
    arr_2d = ccreate(args.N)
    arr_cropped = crop_array(arr_2d, 1,1,2,2)
    pad_arr = padd(arr_cropped)
    c_arr = cconcatenate(pad_arr)
    print("Original array:")
    print(arr_2d)
    print("\nCropped array:")
    print(arr_cropped)
    print("\nPadded array:")
    print(pad_arr)
    print("\nConcatenated array: shape = {}".format(c_arr.shape))
    print(c_arr)
    
