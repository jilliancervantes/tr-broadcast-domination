# (t,r) Broadcast Domination Numbers and Densities of the Truncated Square Tiling Graph

For a pair of positive integer parameters $(t,r)$, a subset $T$ of vertices of a graph $G$ is said to $(t,r)$ broadcast dominate a graph 
$G$ if, for any vertex $u$ in $G$, the sum of the difference between $t$ and the distance from $u$ to all vertices $v\in T$ is greater than or equal to $r$. 
This can be interpreted as each vertex $v$ of $T$ sending $\max(t-\text{d}(u,v),0)$ signal to vertices within a distance of $t-1$ away from $v$. The signal is additive and we ask that every vertex of the graph received a minimum of $r$ worth of signal from all vertices in $T$.
For a finite graph the smallest cardinality among all $(t,r)$ broadcast dominating sets of a graph is called the $(t,r)$ broadcast domination number.
We remark that the $(2,1)$ broadcast domination number is the domination number and the $(t,1)$ (for $t\geq 1$) is distance domination number of a graph.

We study
a family of graphs that arise as a finite subgraph of the truncated square titling, which utilizes regular squares and octagons to tile the Euclidean plane. 
For positive integers $m$ and $n$, we let $H_{m,n}$ be the graph consisting of $m$ rows of $n$ octagons (cycle graph on $8$ vertices). 

This code utilizes NetworkX to construct truncated square tiling graphs, visualize reception of designated sets of broadcast vertices, determine whether a set of vertices forms a $(t,r)$ broadcast dominating set, and determine whether a specified dominating set cardinality is optimal.
