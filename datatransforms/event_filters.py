import numpy as np

# Pixels to be removed:  (x, y)
NASL_failed_pixels = [  [116,  47],
                        [214,  58],
                        [174,   7],
                        [ 70,  95],
                        [188,  75],
                        [160,  73],
                        [ 64,  18],
                        [175,  18],
                        [120,  77],
                        [122,  88]]

def remove_NASL_failed_pixels(data):
    # Filter out pixels from NASL dataset that generate a lot of events
    # (most likely due to sensor failure)
    # data: torch_geometric.data.Data object
    # Returns: torch_geometric.data.Data object
    #
    # The pixels are listed in NASL_failed_pixels
    
    return get_indices(data.pos[:,:2], NASL_failed_pixels)
    
    
    
    
def get_indices(A,B):
    # Assuming you have 2D arrays A and B of size [N, p] and [M, p] respectively
    # A and B are the two arrays you want to compare

    A = np.array(A, dtype=np.int_)
    B = np.array(B, dtype=np.int_)

    assert A.shape[1] == B.shape[1], f'Arrays must have the same number of columns. A has {A.shape[1]} columns, B has {B.shape[1]} columns.'

    # Reshape the arrays to [N, 1, 2] and [1, M, 2] to enable broadcasting
    A_expanded = A[:, None, :]
    B_expanded = B[None, :, :]

    # Use broadcasting to perform element-wise comparison
    # result will be a boolean tensor of shape [N, M]
    matches = (A_expanded == B_expanded).all(axis=-1)

    # Find the rows in A that have at least one match with B
    indices = np.invert(np.any(matches, axis=1))

    return indices