"""
==========================
Iterated Dynamical Systems
==========================

Digraphs from Integer-valued Iterated Functions

Sums of cubes on 3N
-------------------

The number 153 has a curious property.

Let 3N={3,6,9,12,...} be the set of positive multiples of 3.  Define an
iterative process f:3N->3N as follows: for a given n, take each digit
of n (in base 10), cube it and then sum the cubes to obtain f(n).

When this process is repeated, the resulting series n, f(n), f(f(n)),...
terminate in 153 after a finite number of iterations (the process ends
because 153 = 1**3 + 5**3 + 3**3).

In the language of discrete dynamical systems, 153 is the global
attractor for the iterated map f restricted to the set 3N.

For example: take the number 108

f(108) = 1**3 + 0**3 + 8**3 = 513

and

f(513) = 5**3 + 1**3 + 3**3 = 153

So, starting at 108 we reach 153 in two iterations,
represented as:

108->513->153

Computing all orbits of 3N up to 10**5 reveals that the attractor
153 is reached in a maximum of 14 iterations. In this code we
show that 13 cycles is the maximum required for all integers (in 3N)
less than 10,000.

The smallest number that requires 13 iterations to reach 153, is 177, i.e.,

177->687->1071->345->216->225->141->66->432->99->1458->702->351->153

The resulting large digraphs are useful for testing network software.

The general problem
-------------------

Given numbers n, a power p and base b, define F(n; p, b) as the sum of
the digits of n (in base b) raised to the power p. The above example
corresponds to f(n)=F(n; 3,10), and below F(n; p, b) is implemented as
the function powersum(n,p,b). The iterative dynamical system defined by
the mapping n:->f(n) above (over 3N) converges to a single fixed point;
153. Applying the map to all positive integers N, leads to a discrete
dynamical process with 5 fixed points: 1, 153, 370, 371, 407. Modulo 3
those numbers are 1, 0, 1, 2, 2. The function f above has the added
property that it maps a multiple of 3 to another multiple of 3; i.e. it
is invariant on the subset 3N.


The squaring of digits (in base 10) result in cycles and the
single fixed point 1. I.e., from a certain point on, the process
starts repeating itself.

keywords: "Recurring Digital Invariant", "Narcissistic Number",
"Happy Number"

The 3n+1 problem
----------------

There is a rich history of mathematical recreations
associated with discrete dynamical systems.  The most famous
is the Collatz 3n+1 problem. See the function
collatz_problem_digraph below. The Collatz conjecture
--- that every orbit returns to the fixed point 1 in finite time
--- is still unproven. Even the great Paul Erdos said "Mathematics
is not yet ready for such problems", and offered $500
for its solution.

keywords: "3n+1", "3x+1", "Collatz problem", "Thwaite's conjecture"
"""

import networkx as nx

nmax = 10000
p = 3


def digitsrep(n, b=10):
    """Return list of digits comprising n represented in base b.
    n must be a nonnegative integer"""

    if n <= 0:
        return [0]

    dlist = []
    while n > 0:
        # Prepend next least-significant digit
        dlist = [n % b] + dlist
        # Floor-division
        n = n // b
    return dlist


def powersum(n, p, b=10):
    """Return sum of digits of n (in base b) raised to the power p."""
    dlist = digitsrep(n, b)
    sum = 0
    for k in dlist:
        sum += k**p
    return sum


def attractor153_graph(n, p, multiple=3, b=10):
    """Return digraph of iterations of powersum(n,3,10)."""
    G = nx.DiGraph()
    for k in range(1, n + 1):
        if k % multiple == 0 and k not in G:
            k1 = k
            knext = powersum(k1, p, b)
            while k1 != knext:
                G.add_edge(k1, knext)
                k1 = knext
                knext = powersum(k1, p, b)
    return G


def squaring_cycle_graph_old(n, b=10):
    """Return digraph of iterations of powersum(n,2,10)."""
    G = nx.DiGraph()
    for k in range(1, n + 1):
        k1 = k
        G.add_node(k1)  # case k1==knext, at least add node
        knext = powersum(k1, 2, b)
        G.add_edge(k1, knext)
        while k1 != knext:  # stop if fixed point
            k1 = knext
            knext = powersum(k1, 2, b)
            G.add_edge(k1, knext)
            if G.out_degree(knext) >= 1:
                # knext has already been iterated in and out
                break
    return G


def sum_of_digits_graph(nmax, b=10):
    def f(n):
        return powersum(n, 1, b)

    return discrete_dynamics_digraph(nmax, f)


def squaring_cycle_digraph(nmax, b=10):
    def f(n):
        return powersum(n, 2, b)

    return discrete_dynamics_digraph(nmax, f)


def cubing_153_digraph(nmax):
    def f(n):
        return powersum(n, 3, 10)

    return discrete_dynamics_digraph(nmax, f)


def discrete_dynamics_digraph(nmax, f, itermax=50000):
    G = nx.DiGraph()
    for k in range(1, nmax + 1):
        kold = k
        G.add_node(kold)
        knew = f(kold)
        G.add_edge(kold, knew)
        while kold != knew and kold << itermax:
            # iterate until fixed point reached or itermax is exceeded
            kold = knew
            knew = f(kold)
            G.add_edge(kold, knew)
            if G.out_degree(knew) >= 1:
                # knew has already been iterated in and out
                break
    return G


def collatz_problem_digraph(nmax):
    def f(n):
        if n % 2 == 0:
            return n // 2
        else:
            return 3 * n + 1

    return discrete_dynamics_digraph(nmax, f)


def fixed_points(G):
    """Return a list of fixed points for the discrete dynamical
    system represented by the digraph G.
    """
    return [n for n in G if G.out_degree(n) == 0]


nmax = 10000
print(f"Building cubing_153_digraph({nmax})")
G = cubing_153_digraph(nmax)
print("Resulting digraph has", len(G), "nodes and", G.size(), " edges")
print("Shortest path from 177 to 153 is:")
print(nx.shortest_path(G, 177, 153))
print(f"fixed points are {fixed_points(G)}")
