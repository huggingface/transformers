cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) nogil:
  cdef int x
  cdef int y
  cdef float v_prev
  cdef float v_cur
  cdef float tmp
  cdef int index = t_x - 1

  for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
      if x == y:
        v_cur = max_neg_val
      else:
        v_cur = value[y-1, x]
      if x == 0:
        if y == 0:
          v_prev = 0.
        else:
          v_prev = max_neg_val
      else:
        v_prev = value[y-1, x-1]
      value[y, x] += max(v_prev, v_cur)

  for y in range(t_y - 1, -1, -1):
    path[y, index] = 1
    if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
      index = index - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) nogil:
  cdef int b = paths.shape[0]
  cdef int i
  for i in prange(b, nogil=True):
    maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])