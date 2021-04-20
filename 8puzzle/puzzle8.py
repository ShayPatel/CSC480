from collections import deque
import heapq
import streamlit as st
import pandas as pd
from streamlit.caching import _get_output_hash
import streamlit.components.v1 as components

previous_states = set()

def print_state(state:str):
    """Prints the state in a 3x3 board

    Parameters
    ----------
    state : str
        string representation of the board state
    """
    arr = [state[a:a+3] for a in (0,3,6)]
    for a in arr:
        print(" ".join(a))


def successor(state:str):
    """Generates the next moves from a given state.
    The function requires a global set called "previous_state".
    New states found in the global set are not returned.

    Parameters
    ----------
    state : str
        String representation of the board state

    Returns
    -------
    tuple[int,string]
        A tuple containing the cost of the move and the new state
    """
    index = state.index("0")
    i = index//3
    j = index%3

    next_moves = []
    #move up
    if i > 0:
        #convert the string to a list to make it mutable
        temp_state = list(state)
        
        #switch the values in the list
        temp_state[index], temp_state[index-3] = temp_state[index-3], temp_state[index]
        
        #the cost of the move will be the value of the new item at index
        cost = int(temp_state[index])
        
        #join the new state to get the string representation back
        new_state = "".join(temp_state)
        
        #check if the state was previously visited
        if new_state not in previous_states:
            #keep the cost the first item in the tuple to be used in a heapq object
            next_moves.append((cost,new_state,"UP"))
    #move down
    if i < 2:
        temp_state = list(state)
        temp_state[index], temp_state[index+3] = temp_state[index+3], temp_state[index]
        cost = int(temp_state[index])
        new_state = "".join(temp_state)
        if new_state not in previous_states:
            next_moves.append((cost,new_state,"DOWN"))
    #move left
    if j > 0:
        temp_state = list(state)
        temp_state[index], temp_state[index-1] = temp_state[index-1], temp_state[index]
        cost = int(temp_state[index])
        new_state = "".join(temp_state)
        if new_state not in previous_states:
            next_moves.append((cost,new_state,"LEFT"))
    #move right
    if j < 2:
        temp_state = list(state)
        temp_state[index], temp_state[index+1] = temp_state[index+1], temp_state[index]
        cost = int(temp_state[index])
        new_state = "".join(temp_state)
        if new_state not in previous_states:
            next_moves.append((cost,new_state,"RIGHT"))
    
    return next_moves


class Node:
    """A class that can be used to create the search tree
    """
    def __init__(self,state:str,total_cost:int,path_length:int,move:str=None,predecessor=None):
        """Constructor for the Node.

        Parameters
        ----------
        state : str
            String representation of the board state
        total_cost : int
            total cost of the moves leading to the node
        path_length : int
            number of nodes in the path
        move : str, optional
            the move taken to reach the node, by default None
        predecessor : Node, optional
            the predecessor node, by default None
        """
        #current state of the node
        self.state = state
        
        #total cost to get to the node
        self.total_cost = total_cost
        
        #depth of the node in the search tree with root as 0
        self.path_length = path_length
        
        #the move taken to reach the node from the previous node
        self.move = move
        
        #the predecessor Node object
        #traverse the predecessor links to get the solution
        self.predecessor = predecessor
        
    #need lt function to allow for heap pop
    def __lt__(self,n):
        return self.total_cost < n.total_cost


def get_solution(n:Node):
    """Generate the solution path given the solution node

    Parameters
    ----------
    n : Node
        the solution node

    Returns
    -------
    List[Node]
        a list of nodes that make up the solution
    """
    #append all the connected nodes in the path to the list
    #uncomment the last append to include the root
    solution = []
    while n.predecessor:
        solution.append(n)
        n = n.predecessor
    #solution.append(n)
    return solution


def h_count(state:str,goal:str):
    """The count of misplaced tiles

    Parameters
    ----------
    state : str
        String representation of the board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    int
        the number of misplaced tiles
    """
    return sum([a==b for a,b in zip(state,goal)])


def h_dist(state:str,goal:str):
    """Calculates the manhattan distance between the given state and the goal state

    Parameters
    ----------
    state : str
        String representation of the board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    int
        the manhattan distance of the given state and the goal state
    """
    dist = 0
    for index,char in enumerate(state):
        i1 = index//3
        j1 = index%3
        
        index_g = goal.index(char)
        i2 = index_g//3
        j2 = index_g%3
        
        dist += abs(i1-i2)+abs(j1-j2)
    return dist


def h_dist_cost(state:str,goal:str):
    """Modified manhattan distance that multiplies the distance of the tile with the cost of moving the tile

    Parameters
    ----------
    state : str
        String representation of the board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    int
        modified manhattan distance
    """
    dist = 0
    for index,char in enumerate(state):
        i1 = index//3
        j1 = index%3
        
        index_g = goal.index(char)
        i2 = index_g//3
        j2 = index_g%3
        
        dist += int(char) * (abs(i1-i2)+abs(j1-j2))
    return dist


def BFS(root:str,goal:str):
    """breadth first search strategy

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    Node
        the solution node
    int
        the maximum size of the queue
    int
        the total number of nodes expanded
    """
    max_q_size = 0
    #testing var for queue size
    q_size = 1
    #testing var for number of nodes popped
    time = 0
    
    #dumb idea using a global set. have to clear each time i run.
    #should I refactor the code to make it a function level set? would probably have to pass it as an arg each time though
    previous_states.clear()
    
    #using a queue to store the ordering
    q = deque()
    
    #add the root to the queue
    q.append(Node(root,0,0))
    
    while q:
        if q_size > max_q_size:
            max_q_size = q_size
        
        #pop the top of the queue to get the current item to work on
        #not pop, use popleft since we are appending right
        current = q.popleft()
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        time += 1
        #check if at the goal
        if current.state == goal:
            return current, max_q_size, time
        else:
            #add the state to the checked states set
            previous_states.add(current.state)
            
            #get the successors
            children = successor(current.state)
            for c in children:
                q.append(Node(c[1], current.total_cost + c[0], current.path_length + 1, c[2], current))
                q_size += 1
            

#getting same results as BFS. why?
#resolved. was appending to wrong side in while loop
def DFS(root:str,goal:str):
    """depth first search strategy

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    Node
        the solution node
    int
        the maximum size of the queue
    int
        the total number of nodes expanded
    """
    max_q_size = 0
    q_size = 1
    time = 0
    
    previous_states.clear()
    
    q = deque()
    
    #now using stack instead of queue, so append
    q.append(Node(root,0,0))
    
    while(q):
        if q_size > max_q_size:
            max_q_size = q_size
        
        #stack so pop from same side as append
        current = q.pop()
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        time += 1
        
        if current.state == goal:
            return current, max_q_size, time
        else:
            previous_states.add(current.state)
            
            children = successor(current.state)
            for c in children:
                #stupid mistake. was using append left before, not append. Should be fixed now.
                q.append(Node(
                    c[1], current.total_cost + c[0], current.path_length + 1, c[2], current
                ))
                q_size += 1


#cloned from DFS function
def DFS_limited(root:str,goal:str,limit=50):
    """depth first search strategy that limits the maximum depth of the search tree

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state
    limit : int, optional
        the maximum depth of the search tree, by default 50

    Returns
    -------
    Node
        the solution node
    int
        the maximum size of the queue
    int
        the total number of nodes expanded
    """
    max_q_size = 0
    q_size = 1
    time = 0
    
    previous_states.clear()
    
    q = deque()
    
    #now using stack instead of queue, so append
    q.append(Node(root,0,0))
    
    while(q):
        if q_size > max_q_size:
            max_q_size = q_size
        
        #stack so pop from same side as append
        current = q.pop()
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        time += 1
        
        if current.state == goal:
            return current, max_q_size, time
        else:
            previous_states.add(current.state)
            
            if current.path_length > limit:
                continue
            else:
                children = successor(current.state)
                for c in children:
                    q.append(Node(
                        c[1], current.total_cost + c[0], current.path_length + 1, c[2], current
                    ))
                    q_size += 1


def uniform_cost(root:str,goal:str):
    """uniform cost search strategy

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state

    Returns
    -------
    Node
        the solution node
    int
        the maximum queue size
    int
        the total number of noded expanded
    """
    #have to use 2nd q size var
    #q size keeps track of current queue size
    #do max size check at each loop iteration
    max_q_size = 0
    q_size = 1
    time = 0
    
    previous_states.clear()
    
    #use heap for queue
    #heap tracks the total cost of the node traversal
    heap = []
    heapq.heappush(heap,(0,Node(root,0,0)))
    
    while(heap):
        if q_size > max_q_size:
            max_q_size = q_size
            
        cost, current = heapq.heappop(heap)
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        #add time after repeat check since we are not expanding that node
        time += 1
        
        if current.state == goal:
            return current, max_q_size, time
        else:
            #in this case, the expanded nodes for a state is the closest to the root
            #the node will either:
            #1 - expanded early but not part of the solution path. In which case this will prevent further search depth
            #2 - part of the solution path
            #3 - not part of the solution path and will never be expanded
            #allows us to keep the set if we add the expanded nodes to it
            previous_states.add(current.state)
            
            children = successor(current.state)

            for c in children:
                #bit of a change here with the heap. Have to add tuple of (total cost,node)
                heapq.heappush(heap,(current.total_cost + c[0], Node(
                    c[1], current.total_cost + c[0], current.path_length + 1, c[2], current
                )))
                #queue size here is basically all the items in the heap.
                #some states will have repetition, but repeat checking would have linear runtime so not worth doing.
                q_size += 1


#clone from uniform function. change the value added to the heap tuple
#TODO: the solution seems fine, but the space and time are too much
#I think the heuristic function produces too many of the same results (not enough range)
def best_first(root:str,goal:str,h_func):
    """best first search strategy given a heuristic function

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state
    h_func : Callable
        the heuristic function with signature of (str,str)

    Returns
    -------
    Node
        the solution node
    int
        the maximum queue size
    int
        the total number of nodes expanded
    """
    #use a function as an arg to avoid adding the logic in the search function
    max_q_size = 0
    q_size = 1
    time = 0
    
    previous_states.clear()
    
    heap = []
    heapq.heappush(heap,(0,Node(root,0,0)))
    
    while(heap):
        if q_size > max_q_size:
            max_q_size = q_size
            
        cost, current = heapq.heappop(heap)
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        time += 1
        
        if current.state == goal:
            return current, max_q_size, time
        else:
            #can still use the hash set as the state has a set 
            previous_states.add(current.state)
            
            children = successor(current.state)
            
            for c in children:
                #the heap priority is changed from total cost to the h_func given as an arg
                #h_func gives the heuristic from the child state to the goal
                heapq.heappush(heap,(h_func(c[1],goal), Node(
                    c[1], current.total_cost + c[0], current.path_length + 1, c[2], current
                )))
                q_size += 1


def a_star(root:str,goal:str,h_func):
    """a* search strategy given a heuristic function

    Parameters
    ----------
    root : str
        String representation of the root board state
    goal : str
        String representation of the goal board state
    h_func : Callable
        the heuristic function with signature of (str,str)

    Returns
    -------
    Node
        the solution node
    int
        the maximum queue size
    int
        the total number of nodes expanded
    """
    max_q_size = 0
    q_size = 1
    time = 0
     
    previous_states.clear()
    
    heap = []
    heapq.heappush(heap,(0,Node(root,0,0)))
    
    while(heap):
        if q_size > max_q_size:
            max_q_size = q_size
        
        cost, current = heapq.heappop(heap)
        q_size -= 1
        
        if current.state in previous_states:
            continue
        
        time += 1
        
        if current.state == goal:
            return current, max_q_size, time
        else:
            previous_states.add(current.state)
            
            children = successor(current.state)
            
            for c in children:
                heapq.heappush(heap,(
                    #cost + h_func
                    current.total_cost + c[0] + h_func(c[1],goal), 
                    Node(
                        c[1], current.total_cost + c[0], current.path_length + 1, c[2], current
                    )
                ))
                q_size += 1


def show_results(start:str,goal:str,search,show=True,**kwargs):
    """Wrapper function to run the search function and show the results

    Parameters
    ----------
    start : str
        String representation of the root board state
    goal : str
        String representation of the goal board state
    search : Callable
        the search function
    show : bool, optional
        Option to show the solution steps, by default True

    kwargs
    ------
    Specific keyword args for the search function
    """
    found, q_size, time = search(start,goal,**kwargs)
    
    print(f"Time: {time}")
    print(f"Space: {q_size}")
    print(f"Cost: {found.total_cost}")
    print(f"Path length: {found.path_length}")
    print()

    if show:
        solution = get_solution(found)

        print("starting state")
        print_state(start)
        for i in reversed(solution):
            print()
            print(f"{i.move}, total_cost: {i.total_cost}")
            print_state(i.state)


def summary(root:str,goal:str,limit=100):
    print("BFS")
    show_results(root, goal, BFS,show=False)

    print("DFS")
    show_results(root, goal, DFS,show=False)

    #limit = 100
    try:
        print(f"DFS limited to depth {limit}")
        show_results(root, goal, DFS_limited,show=False,limit=limit)
    except Exception:
        print(f"unable to find solution with depth limit: {limit}")
        print()
        
    print("Uniform cost")
    show_results(root, goal, uniform_cost,show=False)

    print("Best first with misplaced count")
    show_results(root, goal, best_first, h_func=h_count,show=False)

    print("Best first with manhattan distance")
    show_results(root, goal, best_first, h_func=h_dist,show=False)

    print("a* with misplaced count")
    show_results(root, goal, a_star, h_func=h_count,show=False)

    print("a* with manhattan distance")
    show_results(root, goal, a_star, h_func=h_dist,show=False)

    print("a* with manhattan distance*cost")
    show_results(root, goal, a_star, h_func=h_dist_cost,show=False)


def st_state(state:str):
    """Creates an html string to create a board for the given state

    Parameters
    ----------
    state : str
        string representation of the board state
    """

    style = """
        <style>
            .chess-board { border-spacing: 0; border-collapse: collapse; }
            .chess-board th { padding: .5em; }
            .chess-board td { border: 1px solid; width: 2em; height: 2em; }
            .chess-board .light { background: #eee; }
            .chess-board .dark { background: #000; }
        </style>
        """
    table = f"""
        <table class="chess-board">
            <tbody>
                <tr>
                    <td class="{"light" if int(state[0])>0 else "dark"}">{state[0]}</td>
                    <td class="{"light" if int(state[1])>0 else "dark"}">{state[1]}</td>
                    <td class="{"light" if int(state[2])>0 else "dark"}">{state[2]}</td>
                </tr>
                <tr>
                    <td class="{"light" if int(state[3])>0 else "dark"}">{state[3]}</td>
                    <td class="{"light" if int(state[4])>0 else "dark"}">{state[4]}</td>
                    <td class="{"light" if int(state[5])>0 else "dark"}">{state[5]}</td>
                </tr>
                <tr>
                    <td class="{"light" if int(state[6])>0 else "dark"}">{state[6]}</td>
                    <td class="{"light" if int(state[7])>0 else "dark"}">{state[7]}</td>
                    <td class="{"light" if int(state[8])>0 else "dark"}">{state[8]}</td>
                </tr>
            </tbody>
        </table>
        """

    return style + table


def st_results(start:str,goal:str,search,show=True,output_limit=50,**kwargs):
    """Wrapper funtion to generate a streamlit area to display the output of a search

    Parameters
    ----------
    start : str
        String representation of the start board state
    goal : str
        String representation of the goal board state
    search : Callable
        search function
    show : bool, optional
        option to show the board output, by default True
    output_limit : int, optional
        the maximum number of board outputs allowed, by default 50
    """
    try:
        found, q_size, time = search(start,goal,**kwargs)
    except Exception:
        st.error("Unable to find a solution")
        return

    results = {
        "Time": time,
        "Space": q_size,
        "Cost": found.total_cost,
        "Path length": found.path_length
    }

    left,right = st.beta_columns(2)

    df = pd.DataFrame({
        "Field": results.keys(),
        "Value": results.values()
    }).set_index("Field")
    left.table(df)

    if show:
        if found.path_length > output_limit:
            right.error(f"Output hidden. The path length exceeds output limit of {output_limit}")
        else:
            with right.beta_expander("States") as exp:
                solution = get_solution(found)
                st.info("starting state")
                st.markdown(
                    st_state(start),
                    unsafe_allow_html=True
                )

                #print_state(start)
                for i in reversed(solution):
                    st.info(f"{i.path_length}. {i.move}, total_cost: {i.total_cost}")
                    st.markdown(
                        st_state(i.state),
                        unsafe_allow_html=True
                    )


def st_summary(start:str,goal:str,show=False,output_limit=50,dfs_limit=40):
    """Creates a streamlit summary page of all the search output data

    Parameters
    ----------
    start : str
        String representation of the start board state
    goal : str
        String representation of the goal board state
    show : bool, optional
        option to show the board outputs, by default False
    output_limit : int, optional
        the maximum number of board outputs allowed, by default 50
    dfs_limit : int, optional
        the maximum depth of the dfs search, by default 40
    """
    st.success("BFS")
    st_results(start, goal, BFS, show=show, output_limit=output_limit)

    st.success("DSF unrestricted")
    st_results(start, goal, DFS, show=show, output_limit=output_limit)

    st.success(f"DFS limited to depth {dfs_limit}")
    try:
        st_results(start, goal, DFS_limited, show=show, limit=dfs_limit, output_limit=output_limit)
    except Exception:
        st.error(f"unable to find solution with depth limit: {dfs_limit}")
        
    st.success("Uniform cost")
    st_results(start, goal, uniform_cost, show=show, output_limit=output_limit)

    st.success("Best first with misplaced count")
    st_results(start, goal, best_first, h_func=h_count, show=show, output_limit=output_limit)

    st.success("Best first with manhattan distance")
    st_results(start, goal, best_first, h_func=h_dist, show=show, output_limit=output_limit)

    st.success("Best first with manhattan distance * cost")
    st_results(start, goal, best_first, h_func=h_dist_cost, show=show, output_limit=output_limit)

    st.success("a* with misplaced count")
    st_results(start, goal, a_star, h_func=h_count, show=show, output_limit=output_limit)

    st.success("a* with manhattan distance")
    st_results(start, goal, a_star, h_func=h_dist, show=show, output_limit=output_limit)

    st.success("a* with manhattan distance \* cost")
    st_results(start, goal, a_star, h_func=h_dist_cost, show=show, output_limit=output_limit)


def st_compare_result(col,start:str,goal:str,search,show=True,output_limit=50,**kwargs):
    """Wrapper function to create a column with the search strategy output

    Parameters
    ----------
    col : Streamlit.beta_column
        streamlit column object
    start : str
        String representation of the start board state
    goal : str
        String representation of the goal board state
    search : Callable
        the search function
    show : bool, optional
        option to show the boards, by default True
    output_limit : int, optional
        the maximum number of boards to output, by default 50
    """
    try:
        found, q_size, time = search(start,goal,**kwargs)
    except Exception:
        col.error("Unable to find a solution")
        return

    results = {
        "Time": time,
        "Space": q_size,
        "Cost": found.total_cost,
        "Path length": found.path_length
    }

    df = pd.DataFrame({
        "Field": results.keys(),
        "Value": results.values()
    }).set_index("Field")
    col.table(df)

    if show:
        if found.path_length > output_limit:
            col.error(f"Output hidden. The path length exceeds output limit of {output_limit}")
        else:
            with col.beta_expander("States") as exp:
                solution = get_solution(found)
                st.info("starting state")
                st.markdown(
                    st_state(start),
                    unsafe_allow_html=True
                )

                #print_state(start)
                for i in reversed(solution):
                    st.info(f"{i.path_length}. {i.move}, total cost: {i.total_cost}")
                    st.markdown(
                        st_state(i.state),
                        unsafe_allow_html=True
                    )


def st_compare(start:str,goal:str,strategies:list,output_limit=50,dfs_limit=40,show=True):
    """Creates streamlit columns to compare the data of selected search strategies

    Parameters
    ----------
    start : str
        String representation of the start state
    goal : str
        String representation of the goal state
    strategies : list
        list of search strategies selected
    output_limit : int, optional
        the maximum number of boards to show in the output, by default 50
    dfs_limit : int, optional
        the maximum depth of the depth first search strategy, by default 40
    show : bool, optional
        option to show the boards, by default True
    """
    cols = st.beta_columns(len(strategies))
    
    for i, strat in enumerate(strategies):
        if strat == "BFS":
            cols[i].success("BFS")
            st_compare_result(cols[i],start, goal, BFS, show=show, output_limit=output_limit)
        elif strat == "DFS":
            cols[i].success("DSF unrestricted")
            st_compare_result(cols[i],start, goal, DFS, show=show, output_limit=output_limit)
        elif strat == "DFS limited":
            cols[i].success(f"DFS limited to depth {dfs_limit}")
            try:
                st_compare_result(cols[i],start, goal, DFS_limited, show=show, limit=dfs_limit, output_limit=output_limit)
            except Exception:
                cols[i].error(f"unable to find solution with depth limit: {dfs_limit}")
        elif strat == "Uniform cost":
            cols[i].success("Uniform cost")
            st_compare_result(cols[i],start, goal, uniform_cost, show=show, output_limit=output_limit)
        elif strat == "Best first with misplaced tiles":
            cols[i].success("Best first with misplaced count")
            st_compare_result(cols[i],start, goal, best_first, h_func=h_count, show=show, output_limit=output_limit)
        elif strat == "Best first with manhattan distance":
            cols[i].success("Best first with manhattan distance")
            st_compare_result(cols[i],start, goal, best_first, h_func=h_dist, show=show, output_limit=output_limit)
        elif strat == "Best first with modified manhattan distance":
            cols[i].success("Best first with manhattan distance * cost")
            st_compare_result(cols[i],start, goal, best_first, h_func=h_dist_cost, show=show, output_limit=output_limit)
        elif strat == "a* with misplaced tiles":
            cols[i].success("a* with misplaced count")
            st_compare_result(cols[i],start, goal, a_star, h_func=h_count, show=show, output_limit=output_limit)
        elif strat == "a* with manhattan distance":
            cols[i].success("a* with manhattan distance")
            st_compare_result(cols[i],start, goal, a_star, h_func=h_dist, show=show, output_limit=output_limit)
        elif strat == "a* with modified manhattan distance":
            cols[i].success("a* with manhattan distance \* cost")
            st_compare_result(cols[i],start, goal, a_star, h_func=h_dist_cost, show=show, output_limit=output_limit)



ouput_limit = st.sidebar.select_slider("Output limit",range(101),value=50)
dfs_limit = st.sidebar.select_slider("DFS depth limit",range(101),value=40)

start_option = st.sidebar.radio("Preset start states",["None","Easy","Medium","Hard"])
goal_option = st.sidebar.radio("Preset goal states",["None","Default"])
start_states = {
    "None": "",
    "Easy": "134862705",
    "Medium": "281043765",
    "Hard": "567408321"
}
goal_states = {
    "None": "",
    "Default": "123804765"
}


start = st.sidebar.text_input("Start state",start_states[start_option])
goal = st.sidebar.text_input("Goal state",goal_states[goal_option])

start = start.replace(" ","").strip()
goal = goal.replace(" ","").strip()


view_option = st.sidebar.radio("Views",["Summary","Compare"])

if view_option == "Summary":
    if len(start) == 9 and len(goal) == 9:
        #st_results(s1, g, a_star, h_func=h_dist_cost)
        st_summary(start,goal,show=True,output_limit=ouput_limit,dfs_limit=dfs_limit)
elif view_option == "Compare":
    strategies = ["BFS","DFS","DFS limited","Uniform cost","Best first with misplaced tiles","Best first with manhattan distance","Best first with modified manhattan distance","a* with misplaced tiles","a* with manhattan distance","a* with modified manhattan distance"]
    search_options = st.sidebar.multiselect("Search strategies",strategies)
    if not search_options:
        st.warning("Select search strategies")
    elif len(start) == 9 and len(goal) == 9:
        st_compare(start,goal,search_options,ouput_limit,dfs_limit)
    else:
        st.warning("Enter valid state values")