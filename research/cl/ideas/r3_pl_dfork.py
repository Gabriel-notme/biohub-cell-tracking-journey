"""Critic-B probe (pipeline angle): re-apply dfork (whole-movie double-fork resolution) at the very END of P14, i.e. after
relink/edge_link/term_trim/long_link, which can join lineages and create new nested forks that the early dfork never saw.
GT-free; only removes a fork branch edge."""
import sys
sys.path.insert(0, '/workspace/p56stage')


def apply(nodes, edges, K=35):
    import dfork
    ne, st = dfork.resolve(nodes, edges, K=K)
    return nodes, ne, {'final_dfork_removed': st['dfork_removed']}
