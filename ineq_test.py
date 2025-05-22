from dfa import Node, dfa_to_list, calc_inequality_edges 

if __name__ == '__main__':
    # dfa = Node(
    #     id=0,
    #     a=Node(
    #         id=1,
    #         state=Node.REJ,
    #         a=Node(id=2, state=Node.ACC),
    #         b=Node(id=3, state=Node.REJ)
    #     ),
    #     b=Node(id=4, state=Node.ACC)
    # )

    dfa = Node(
        id=0,
        a=Node(
            id=1,
            state=Node.ACC,
            b=Node(
                id=2,
                a=Node(
                    id=3,
                    a=Node(id=4, state=Node.ACC)
                ),
                b=Node(id=5, state=Node.REJ)
            )
        ),
        b=Node(
            id=6,
            state=Node.REJ,
            b=Node(id=7, state=Node.ACC)
        )
    )

    inequality_edges = calc_inequality_edges(dfa)
    print("ineq edges:")
    for edge in inequality_edges:
        print(edge)