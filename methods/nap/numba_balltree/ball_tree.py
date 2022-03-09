# This code is a BallTree Numba version translated from cython code included here: https://github.com/jakevdp/pyTree

"""
=========
Ball Tree
=========
A ball tree is a data object which speeds up nearest neighbor
searches in high dimensions (see scikit-learn neighbors module
documentation for an overview of neighbor trees). There are many
types of ball trees.  This package provides a basic implementation
in cython.

Implementation Notes
--------------------

A ball tree can be thought of as a collection of nodes.  Each node
stores a centroid, a radius, and the pointers to two child nodes.

* centroid : the centroid of a node is the mean of all the locations
    of points within the node
* radius : the radius of a node is the distance from the centroid
    to the furthest point in the node.
* subnodes : each node has a maximum of 2 child nodes.  The data within
    the parent node is divided between the two child nodes.

In a typical tree implementation, nodes may be classes or structures which
are dynamically allocated as needed.  This offers flexibility in the number
of nodes, and leads to very straightforward and readable code.  It also means
that the tree can be dynamically augmented or pruned with new data, in an
in-line fashion.  This approach generally leads to recursive code: upon
construction, the head node constructs its child nodes, the child nodes
construct their child nodes, and so-on.

The current package uses a different approach: all node data is stored in
a set of numpy arrays which are pre-allocated.  The main advantage of this
approach is that the whole object can be quickly and easily saved to disk
and reconstructed from disk.  This also allows for an iterative interface
which gives more control over the heap, and leads to speed.  There are a
few disadvantages, however: once the tree is built, augmenting or pruning it
is not as straightforward.  Also, the size of the tree must be known from the
start, so there is not as much flexibility in building it.

BallTree Pseudo-code
~~~~~~~~~~~~~~~~~~~~
Because understanding a ball tree is simpler with recursive code, here is some
pseudo-code to show the structure of the main functionality

    # Ball Tree pseudo code

    class Node:
        #class data:
        centroid
        radius
        child1, child2

        #class methods:
        def construct(data):
            centroid = compute_centroid(data)
            radius = compute_radius(centroid, data)

            # Divide the data into two approximately equal sets.
            # This is often done by splitting along a single dimension.
            data1, data2 = divide(data)

            if number_of_points(data1) > 0:
                child1.construct(data1)

            if number_of_points(data2) > 0:
                child2.construct(data2)

        def query(pt, neighbors_heap):
            # compute the minimum distance from pt to any point in this node
            d = distance(point, centroid)
            if d < radius:
                min_distance = 0
            else:
                min_distance = d - radius

            if min_distance > max_distance_in(neighbors_heap):
                # all these points are too far away.  cut off the search here
                return
            elif node_size > 1:
                child1.query(pt, neighbors_heap)
                child2.query(pt, neighbors_heap)


    object BallTree:
        #class data:
        data
        root_node

        #class methods
        def construct(data, num_leaves):
            root_node.construct(data)

        def query(point, num_neighbors):
            neighbors_heap = empty_heap_of_size(num_neighbors)
            root_node.query(point, neighbors_heap)

This certainly is not a complete description, but should give the basic idea
of the form of the algorithm.  The implementation below is much faster than
anything mirroring the pseudo-code above, but for that reason is much more
opaque.  Here's the basic idea:

BallTree Storage
~~~~~~~~~~~~~~~~
The BallTree information is stored using a combination of
"Array of Structures" and "Structure of Arrays" to maximize speed.
Given input data of size ``(n_samples, n_features)``, BallTree computes the
expected number of nodes ``n_nodes`` (see below), and allocates the
following arrays:

* ``data`` : a float array of shape ``(n_samples, n_features)``
    This is simply the input data.  If the input matrix is well-formed
    (contiguous, c-ordered, correct data type) then no copy is needed
* ``idx_array`` : an integer array of size ``n_samples``
    This can be thought of as an array of pointers to the data in ``data``.
    Rather than shuffling around the data itself, we shuffle around pointers
    to the rows in data.
* ``node_centroid_arr`` : a float array of shape ``(n_nodes, n_features)``
    This stores the centroid of the data in each node.
* ``node_info_arr`` : a size-``n_nodes`` array of ``NodeInfo`` structures.
    This stores information associated with each node.  Each ``NodeInfo``
    instance has the following attributes:
    - ``idx_start``
    - ``idx_end`` : ``idx_start`` and ``idx_end`` reference the part of
      ``idx_array`` which point to the data associated with the node.
      The data in node with index ``i_node`` is given by
      ``data[idx_array[idx_start:idx_end]]``
    - ``is_leaf`` : a boolean value which tells whether this node is a leaf:
      that is, whether or not it has children.
    - ``radius`` : a floating-point value which gives the distance from
      the node centroid to the furthest point in the node.

One feature here is that there are no stored pointers from parent nodes to
child nodes and vice-versa.  These pointers are implemented implicitly:
For a node with index ``i``, the two children are found at indices
``2 * i + 1`` and ``2 * i + 2``, while the parent is found at index
``floor((i - 1) / 2)``.  The root node has no parent.

With this data structure in place, the functionality of the above BallTree
pseudo-code can be implemented in a much more efficient manner.
Most of the data passing done in this code uses raw data pointers.
Using numpy arrays would be preferable for safety, but the
overhead of array slicing and sub-array construction leads to execution
time which is several orders of magnitude slower than the current
implementation.

Priority Queue vs Max-heap
~~~~~~~~~~~~~~~~~~~~~~~~~~
When querying for more than one neighbor, the code must maintain a list of
the current k nearest points.  The BallTree code implements this in two ways.

- A priority queue: this is just a sorted list.  When an item is added,
  it is inserted in the appropriate location.  The cost of the search plus
  insert averages O[k].
- A max-heap: this is a binary tree structure arranged such that each node is
  greater than its children.  The cost of adding an item is O[log(k)].
  At the end of the iterations, the results must be sorted: a quicksort is
  used, which averages O[k log(k)].  Quicksort has worst-case O[k^2]
  performance, but because the input is already structured in a max-heap,
  the worst case will not be realized.  Thus the sort is a one-time operation
  with cost O[k log(k)].

Each insert is performed an average of log(N) times per query, where N is
the number of training points.  Because of this, for a single query, the
priority-queue approach costs O[k log(N)], and the max-heap approach costs
O[log(k)log(N)] + O[k log(k)].  Tests show that for sufficiently large k,
the max-heap approach out-performs the priority queue approach by a factor
of a few.  In light of these tests, the code uses a priority queue for
k < 5, and a max-heap otherwise.

Memory Allocation
~~~~~~~~~~~~~~~~~
It is desirable to construct a tree in as balanced a way as possible.
Given a training set with n_samples and a user-supplied leaf_size, if
the points in each node are divided as evenly as possible between the
two children, the maximum depth needed so that leaf nodes satisfy
``leaf_size <= n_points <= 2 * leaf_size`` is given by
``n_levels = 1 + max(0, floor(log2((n_samples - 1) / leaf_size)))``
(with the exception of the special case where ``n_samples < leaf_size``)
For a given number of levels, the number of points in a tree is given by
``n_nodes = 2 ** n_levels - 1``.  Both of these results can be shown
by induction.  Using them, the correct amount of memory can be pre-allocated
for a given ``n_samples`` and ``leaf_size``.
"""

import numpy as np
import torch
import numba

######################################################################
infinity = np.inf


# utility functions: fast max, min, and absolute value
#
@numba.njit
# @cuda.jit('int64(int64, int64)', device=True, debug=True, opt=False)
def dmax(x, y):
    if x >= y:
        return x
    else:
        return y


@numba.njit
def dmax_(x, y):
    if x >= y:
        return x
    else:
        return y


@numba.njit
# @cuda.jit('int64(int64, int64)', device=True, debug=True, opt=False)
def dmin(x, y):
    if x <= y:
        return x
    else:
        return y


@numba.njit
# @cuda.jit('int64(int64)', device=True, debug=True, opt=False)
def dabs(x):
    if x >= 0:
        return x
    else:
        return -x


def hamming_(a, b):
    sum = 0
    for i in range(a.shape[0]):
        sum += a[i] ^ b[i]
    return sum


# wrap = cuda.jit('int64(int64[:], int64[:])', device=True, debug=True, opt=False)
wrap = numba.njit
dist = wrap(hamming_)


@numba.njit
def dist_(a, b):
    sum = 0
    for i in range(a.shape[0]):
        sum += a[i] ^ b[i]
    return sum


@numba.njit
# @cuda.jit('void(int64[:], int64, int64[:])', device=True, debug=True, opt=False)
def stack_push(s, item, stack_info):
    if stack_info[1] >= stack_info[0]:
        raise ValueError("resize")

    s[stack_info[1]] = item
    stack_info[1] = stack_info[1] + 1


@numba.njit
# @cuda.jit('int64(int64[:], int64[:])', device=True, debug=True, opt=False)
def stack_pop(s, stack_info):
    if stack_info[1] == 0:
        raise ValueError("popping empty stack")

    stack_info[1] -= 1
    return s[stack_info[1]]


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


######################################################################
# BallTree class
#
class BallTree(object):
    """
    Ball Tree for fast nearest-neighbor searches :

    BallTree(X, leaf_size=20, p=2.0)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        n_samples is the number of points in the data set, and
        n_features is the dimension of the parameter space.
        Note: if X is a C-contiguous array of doubles then data will
        not be copied. Otherwise, an internal copy will be made.

    leaf_size : positive integer (default = 20)
        Number of points at which to switch to brute-force. Changing
        leaf_size will not affect the results of a query, but can
        significantly impact the speed of a query and the memory required
        to store the built ball tree.  The amount of memory needed to
        store the tree scales as
        2 ** (1 + floor(log2((n_samples - 1) / leaf_size))) - 1
        For a specified ``leaf_size``, a leaf node is guaranteed to
        satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
        the case that ``n_samples < leaf_size``.

    p : distance metric for the BallTree.  ``p`` encodes the Minkowski
        p-distance:
            D = sum((X[i] - X[j]) ** p) ** (1. / p)
        p must be greater than or equal to 1, so that the triangle
        inequality will hold.  If ``p == np.inf``, then the distance is
        equivalent to
            D = max(X[i] - X[j])

    Examples
    --------
    Query for k-nearest neighbors

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        >>> ball_tree = BallTree(X, leaf_size=2)
        >>> dist, ind = ball_tree.query(X[0], n_neighbors=3)
        >>> print ind  # indices of 3 closest neighbors
        [0 3 1]
        >>> print dist  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]

    Pickle and Unpickle a ball tree (using protocol = 2).  Note that the
    state of the tree is saved in the pickle operation: the tree is not
    rebuilt on un-pickling

        >>> import numpy as np
        >>> import pickle
        >>> np.random.seed(0)
        >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
        >>> ball_tree = BallTree(X, leaf_size=2)
        >>> s = pickle.dumps(ball_tree, protocol=2)
        >>> ball_tree_copy = pickle.loads(s)
        >>> dist, ind = ball_tree_copy.query(X[0], k=3)
        >>> print ind  # indices of 3 closest neighbors
        [0 3 1]
        >>> print dist  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]
    """


    def __init__(self, X, leaf_size=20, p=2):
        self.data = np.asarray(X, dtype=np.int64, order='C')

        if X.size == 0:
            raise ValueError("X is an empty array")

        if self.data.ndim != 2:
            raise ValueError("X should have two dimensions")

        if p < 1:
            raise ValueError("p must be greater than or equal to 1")
        self.p = p

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size

        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]

        # determine number of levels in the ball tree, and from this
        # the number of nodes in the ball tree
        self.n_levels = int(np.log2(max(1, (n_samples - 1) / self.leaf_size)) + 1)
        self.n_nodes = (2 ** self.n_levels) - 1

        self.idx_array = np.arange(n_samples, dtype=np.int64)

        self.node_centroid_arr = np.zeros((self.n_nodes, n_features),
                                          dtype=np.int64, order='C')

        self.node_info_arr_start = np.zeros(self.n_nodes,
                                            dtype=np.int64, order='C')

        self.node_info_arr_end = np.zeros(self.n_nodes,
                                          dtype=np.int64, order='C')

        self.node_info_arr_isleaf = np.zeros(self.n_nodes,
                                             dtype=np.int64, order='C')

        self.node_info_arr_radius = np.zeros(self.n_nodes,
                                             dtype=np.int64, order='C')
        build_tree_(self.data, self.idx_array, self.node_centroid_arr, self.node_info_arr_start, self.node_info_arr_end,
                    self.node_info_arr_isleaf, self.node_info_arr_radius, p, n_samples, n_features, self.n_nodes)

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (BallTree,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        return (self.data,
                self.idx_array,
                self.node_centroid_arr,
                self.node_info_arr,
                self.p,
                self.leaf_size,
                self.n_levels,
                self.n_nodes)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.data = state[0]
        self.idx_array = state[1]
        self.node_centroid_arr = state[2]
        self.node_info_arr = state[3]
        self.p = state[4]
        self.leaf_size = state[5]
        self.n_levels = state[6]
        self.n_nodes = state[7]

    def query(self, X, k=1, return_distance=True, threadsperblock=64):  # todo cuda
        """
        query(X, k=1, return_distance=True)

        query the Ball Tree for the k nearest neighbors

        Parameters
        ----------
        X : array-like, last dimension self.dim
            An array of points to query
        k : integer  (default = 1)
            The number of nearest neighbors to return
        return_distance : boolean (default = True)
            if True, return a tuple (d,i)
            if False, return array i

        Returns
        -------
        i    : if return_distance == False
        (d,i) : if return_distance == True

        d : array of doubles - shape: x.shape[:-1] + (k,)
            each entry gives the list of distances to the
            neighbors of the corresponding point
            (note that distances are not sorted)

        i : array of integers - shape: x.shape[:-1] + (k,)
            each entry gives the list of indices of
            neighbors of the corresponding point
            (note that neighbors are not sorted)

        Examples
        --------
        Query for k-nearest neighbors

            >>> import numpy as np
            >>> np.random.seed(0)
            >>> X = np.random.random((10,3))  # 10 points in 3 dimensions
            >>> ball_tree = BallTree(X, leaf_size=2)
            >>> dist, ind = ball_tree.query(X[0], k=3)
            >>> print ind  # indices of 3 closest neighbors
            [0 3 1]
            >>> print dist  # distances to 3 closest neighbors
            [ 0.          0.19662693  0.29473397]
        """
        X = np.asarray(X, dtype=np.int64, order='C')
        if X.ndim != 2:
            raise ValueError("data should be 2 dim")

        if X.shape[-1] != self.data.shape[1]:
            raise ValueError("query data dimension must match BallTree "
                             "data dimension")

        if k > self.data.shape[0]:
            raise ValueError("k must be less than or equal "
                             "to the number of training points")

        # flatten X for iteration
        orig_shape = X.shape
        X = X.reshape((-1, X.shape[-1]))

        n_neighbors = k
        distances = np.zeros((X.shape[0] * n_neighbors),
                             dtype=np.int64)
        idx_array = np.zeros((X.shape[0] * n_neighbors),
                             dtype=np.int64)

        distances[:] = np.iinfo(distances.dtype).max

        stack_nodes = np.zeros(2 * self.n_levels + 1, dtype=np.int64)
        stack_dists = np.zeros(2 * self.n_levels + 1, dtype=np.int64)
        n = X.shape[0]
        blockspergrid = 64
        stack_info = np.zeros(2, dtype=np.int64)
        stack_info[0] = 2 * self.n_levels + 1
        stack_info[1] = 0
        stack_info_2 = np.zeros(2, dtype=np.int64)
        stack_info_2[0] = 2 * self.n_levels + 1
        stack_info_2[1] = 0

        query_(X, n_neighbors, distances, idx_array, stack_info, stack_info_2, stack_nodes,
               stack_dists, self.data, self.idx_array, self.node_centroid_arr,
               self.node_info_arr_start, self.node_info_arr_end,
               self.node_info_arr_isleaf, self.node_info_arr_radius, self.p, self.data.shape[1])

        # deflatten results
        if return_distance:
            return (distances.reshape((orig_shape[:-1]) + (k,)),
                    idx_array.reshape((orig_shape[:-1]) + (k,)))
        else:
            return idx_array.reshape((orig_shape[:-1]) + (k,))


######################################################################
#  This calculates the lower-bound distance between a point and a node
@numba.njit
# @cuda.jit('int64(int64[:], int64[:], int64)', device=True, debug=True, opt=False)
def calc_dist_LB(pt, centroid, radius):
    d = dist(pt, centroid) - radius
    ret = dmax(0, d)
    return ret


######################################################################
# priority queue
#  This is used to keep track of the neighbors as they are found.
#  It keeps the list of neighbors sorted, and inserts each new item
#  into the list.  In this fixed-size implementation, empty elements
#  are represented by infinities.
@numba.njit
# @cuda.jit('int64(int64[:], int64)', device=True, debug=True, opt=False)
def heapqueue_largest(queue, queue_size):
    return queue[queue_size - 1]


@numba.njit
# @cuda.jit('void(int64, int64, int64[:], int64[:], int64)', device=True, debug=True, opt=False)
def heapqueue_insert(val, i_val, queue, idx_array, queue_size):
    i_lower = 0
    i_upper = queue_size - 1

    if val >= queue[i_upper]:
        return
    elif val <= queue[i_lower]:
        i_mid = i_lower
    else:
        while True:
            if (i_upper - i_lower) < 2:
                i_mid = i_lower + 1
                break
            else:
                i_mid = (i_lower + i_upper) // 2

            if i_mid == i_lower:
                i_mid += 1
                break

            if val >= queue[i_mid]:
                i_lower = i_mid
            else:
                i_upper = i_mid

    for i in range(queue_size - 1, i_mid):
        queue[i] = queue[i - 1]
        idx_array[i] = idx_array[i - 1]

    queue[i_mid] = val
    idx_array[i_mid] = i_val


@numba.njit
# @cuda.jit('void(int64[:], int64, int64, int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:, :], int64[:],'
#           ' int64[:, :], int64[:], int64[:], int64[:], int64[:], int64, int64)', device=True, debug=True, opt=False)
def query_one_(pt, outer_i, k, near_set_dist, near_set_indx, stack_info, stack_info_2, stack_nodes, stack_dists, data,
               tree_idx_array, node_centroid_arr, node_info_arr_start, node_info_arr_stop, node_info_arr_isleaf,
               node_info_arr_radius, p, n_features):  # todo cuda

    dist_LB = calc_dist_LB(pt, node_centroid_arr[0],
                           node_info_arr_radius[0])

    stack_push(stack_nodes, 0, stack_info)
    stack_push(stack_dists, dist_LB, stack_info_2)

    while (stack_info[1] > 0):
        i_node = stack_pop(stack_nodes, stack_info)
        dist_LB = stack_pop(stack_dists, stack_info_2)


        # ------------------------------------------------------------
        # Case 1: query point is outside node radius
        if dist_LB >= heapqueue_largest(near_set_dist[outer_i * k:], k):
            continue

        # ------------------------------------------------------------
        # Case 2: this is a leaf node.  Update set of nearby points
        elif node_info_arr_isleaf[i_node]:
            for i in range(node_info_arr_start[i_node], node_info_arr_stop[i_node]):
                dist_pt = dist(pt, data[tree_idx_array[i]])

                if dist_pt < heapqueue_largest(near_set_dist[outer_i * k:], k):
                    heapqueue_insert(dist_pt, tree_idx_array[i],
                                     near_set_dist[outer_i * k:], near_set_indx[outer_i * k:], k)

        # ------------------------------------------------------------
        # Case 3: Node is not a leaf.  Recursively query subnodes
        #         starting with the one whose centroid is closest
        else:
            i1 = 2 * i_node + 1
            i2 = i1 + 1
            dist_LB_1 = calc_dist_LB(pt, node_centroid_arr[i1],
                                     node_info_arr_radius[i1])
            dist_LB_2 = calc_dist_LB(pt, node_centroid_arr[i2],
                                     node_info_arr_radius[i2])

            # append children to stack: last-in-first-out
            if dist_LB_2 <= dist_LB_1:

                stack_push(stack_nodes, i1, stack_info)
                stack_push(stack_dists, dist_LB_1, stack_info_2)

                stack_push(stack_nodes, i2, stack_info)
                stack_push(stack_dists, dist_LB_2, stack_info_2)

            else:

                stack_push(stack_nodes, i2, stack_info)
                stack_push(stack_dists, dist_LB_2, stack_info_2)

                stack_push(stack_nodes, i1, stack_info)
                stack_push(stack_dists, dist_LB_1, stack_info_2)


@numba.njit
# @cuda.jit
def query_(X, n_neighbors, distances, idx_array, stack_info, stack_info_2, stack_nodes, stack_dists, data,
           tree_idx_array, node_centroid_arr, node_info_arr_start, node_info_arr_stop, node_info_arr_isleaf,
           node_info_arr_radius, p, n_features):
    for i, Xi in enumerate(X):
        query_one_(Xi, i, n_neighbors,
                   distances, idx_array, stack_info, stack_info_2, stack_nodes, stack_dists, data,
                   tree_idx_array, node_centroid_arr, node_info_arr_start, node_info_arr_stop, node_info_arr_isleaf,
                   node_info_arr_radius, p, n_features)

        # dist_ptr += n_neighbors
        # idx_ptr += n_neighbors


@numba.njit
def build_tree_(data, idx_array, node_centroid_arr, node_info_arr_start, node_info_arr_stop, node_info_arr_isleaf,
                node_info_arr_radius, p, n_samples, n_features, n_nodes):

    # ------------------------------------------------------------
    # take care of the root node
    node_info_arr_start[0] = 0
    node_info_arr_stop[0] = n_samples
    n_points = n_samples

    # determine Node centroid
    compute_centroid(node_centroid_arr[0], data, idx_array,
                     n_features, n_samples)

    # determine Node radius
    radius = 0
    for i in range(0, n_samples):
        d = dist_(node_centroid_arr[0], data[idx_array[i], :])
        radius = dmax_(radius,
                       d)
    node_info_arr_radius[0] = radius

    # check if this is a leaf
    if n_nodes == 1:
        node_info_arr_isleaf[0] = 1

    else:
        node_info_arr_isleaf[0] = 0

        # find dimension with largest spread
        i_max = find_split_dim(data, idx_array[node_info_arr_start[0]:],
                               n_features, n_points)

        # sort idx_array along this dimension
        partition_indices(data,
                          idx_array[node_info_arr_start[0]:],
                          i_max,
                          n_points // 2,
                          n_features,
                          n_points)

    # ------------------------------------------------------------
    # cycle through all child nodes
    for i_node in range(1, n_nodes):
        i_parent = (i_node - 1) // 2

        if node_info_arr_isleaf[i_parent]:
            raise ValueError("Fatal: parent is a leaf. Memory "
                             "allocation is flawed")

        if i_node < n_nodes // 2:
            node_info_arr_isleaf[i_node] = 0
        else:
            node_info_arr_isleaf[i_node] = 1

        centroid = node_centroid_arr[i_node]

        # find indices for this node
        idx_start = node_info_arr_start[i_parent]
        idx_end = node_info_arr_stop[i_parent]

        if i_node % 2 == 1:
            idx_start = (idx_start + idx_end) // 2
        else:
            idx_end = (idx_start + idx_end) // 2

        node_info_arr_start[i_node] = idx_start
        node_info_arr_stop[i_node] = idx_end

        n_points = idx_end - idx_start

        if n_points == 0:
            raise ValueError("zero-sized node")

        elif n_points == 1:
            # copy this point to centroid
            copy_array(centroid,
                       data[idx_array[idx_start]],
                       n_features)

            # store radius in array
            node_info_arr_radius[i_node] = 0

            # is a leaf
            node_info_arr_isleaf[i_node] = 1

        else:
            # determine Node centroid
            compute_centroid(centroid, data, idx_array[idx_start:],
                             n_features, n_points)

            # determine Node radius
            radius = 0
            for i in range(idx_start, idx_end):
                radius = dmax_(radius,
                               dist_(centroid,
                                     data[idx_array[i]]))
            node_info_arr_radius[i_node] = radius

            if not node_info_arr_isleaf[i_node]:
                # find dimension with largest spread
                i_max = find_split_dim(data, idx_array[idx_start:],
                                       n_features, n_points)

                # sort indices along this dimension
                partition_indices(data,
                                  idx_array[idx_start:],
                                  i_max,
                                  n_points // 2,
                                  n_features,
                                  n_points)


######################################################################
# Helper functions for building and querying

@numba.njit
def copy_array(x, y, n):
    # copy array y into array x
    for i in range(0, n):
        x[i] = y[i]


@numba.njit
def compute_centroid(centroid, data, node_indices, n_features, n_points):
    # `centroid` points to an array of length n_features
    # `data` points to an array of length n_samples * n_features
    # `node_indices` = idx_array + idx_start

    for j in range(0, n_features):
        centroid[j] = 0

    for i in range(0, n_points):
        this_pt = data[node_indices[i]]
        for j in range(0, n_features):
            centroid[j] += this_pt[j]

    for j in range(0, n_features):
        if centroid[j] < n_points / 2:
            centroid[j] = 0
        else:
            centroid[j] = 1


@numba.njit
def find_split_dim(data, node_indices, n_features, n_points):

    j_min = 0
    min_diff = np.inf
    for j in range(0, n_features):

        count_zeros = 0
        count_ones = 0
        for i in range(0, n_points):
            if data[node_indices[i], j] == 0:
                count_zeros += 1
            elif data[node_indices[i], j] == 1:
                count_ones += 1
            else:
                raise RuntimeError("find split")

        diff = abs(count_zeros - count_ones)
        if diff < min_diff:
            min_diff = diff
            j_min = j
    return j_min



@numba.njit
def iswap(arr, i1, i2):
    tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


@numba.njit
def partition_indices(data, node_indices, split_dim, split_index, n_features, n_points):
    # partition_indices will modify the array node_indices between
    # indices 0 and n_points.  Upon return (assuming numpy-style slicing)
    #   data[node_indices[0:split_index], split_dim]
    #     <= data[node_indices[split_index], split_dim]
    # and
    #   data[node_indices[split_index], split_dim]
    #     <= data[node_indices[split_index:n_points], split_dim]
    # will hold.  The algorithm amounts to a partial quicksort
    left = 0
    right = n_points - 1
    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[node_indices[i], split_dim]
            d2 = data[node_indices[right], split_dim]
            if d1 < d2:
                iswap(node_indices, i, midindex)
                midindex += 1
        iswap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


if __name__ == "__main__":
    from sklearn.neighbors import BallTree as sb
    import time

    for known_num in [100, 1000, 10000]:
        for pattern_len in [10, 100, 1000]:
            d = np.array([0] * int(pattern_len * 0.95) + [1] * (pattern_len - int(pattern_len * 0.95)), dtype=np.uint8)
            data_tensor = torch.ones((known_num, pattern_len), device="cuda", dtype=torch.uint8)
            for i in range(known_num):
                np.random.shuffle(d)
                data_tensor[i] = torch.from_numpy(d).cuda()
            data = data_tensor.cpu().numpy()
            b = BallTree(data)
            b2 = sb(data)
            np.random.shuffle(d)
            points_tensor = torch.from_numpy(d).cuda()
            exec_times_numba = np.empty(100)
            exec_times_sklearn = np.empty(100)
            exec_times_tensor = np.empty(100)
            d = d.reshape((1, -1))

            for i in range(100):

                start_time = time.time()
                b.query(d)
                exec_times_numba[i] = time.time() - start_time
            print(f" num: {known_num} len: {pattern_len} numbatime: {exec_times_numba.mean()}")
            for i in range(100):
                start_time = time.time()
                b2.query(d)
                exec_times_sklearn[i] = time.time() - start_time

            print(f" num: {known_num} len: {pattern_len} sklearntime: {exec_times_sklearn.mean()}")
            for i in range(100):
                # b.query(points)
                start_time = time.time()
                # b.query(points)
                for j in range(d.shape[0]):
                    (data_tensor ^ points_tensor[j]).sum(dim=1).min()

                exec_times_tensor[i] = time.time() - start_time
            print(f" num: {known_num} len: {pattern_len} tensortime: {exec_times_tensor.mean()}")

