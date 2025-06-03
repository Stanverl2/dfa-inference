from dfa import generate_prefix_tree, calc_inequality_edges, dfa_to_list, render_graph, render_dfa

if __name__ == "__main__":
    words_in_language = { "aa", "bb" }
    words_not_in_language = { "ab", "ba" }

    
    root = generate_prefix_tree(words_in_language, words_not_in_language)
    
    render_dfa(root, path="/tmp/dfa", view=True)

    dfa_list = dfa_to_list(root)
    
    N = len(dfa_list)
    
    edges = calc_inequality_edges(root)

    render_graph(N, edges, path="/tmp/graph", view=True)
