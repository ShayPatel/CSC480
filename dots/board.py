import streamlit as st
from htbuilder import div, styles
from htbuilder.funcs import rgba, rgb

from random import randint
import numpy as np
from math import inf

from time import sleep,time
import SessionState

sess = SessionState.get(values=None,owners=None,line_count=None,lines=None,human=None,move=None)

size = st.sidebar.slider("board size",2,5,2)
depth = st.sidebar.number_input("max search depth",1,12,6)
num_lines = size*(size+1)

if sess.values is not None:
    values = sess.values
    owners = sess.owners
    line_count = sess.line_count
    lines = sess.lines
    human = sess.human
else:
    #create the random board
    values = np.random.randint(1,6,size=[size,size])
    board_index = np.arange(0,size*size).reshape((size,size))

    owners = np.zeros_like(values)
    line_count = np.zeros_like(values)

    #true means a line is available
    lines = np.ones(2*num_lines,dtype=np.bool)

    sess.values = values
    sess.owners = owners
    sess.line_count = line_count
    sess.lines = lines
    sess.human = True
    sess.move = -1




css = """
        <style>
            #game { display: flex;flex-direction: column; }
            .row { display: flex;flex-direction: row; }
            .dot { height: 4px;width: 4px;background-color: #000; }
            .box { height: 25px;width: 25px;text-align: center; }
            .horizContainer { height: 4px;width: 25px;border: 2px; }
            .vertContainer { height: 25px;width: 4px;border: 2px; }
            #board { display: flex;flex-direction: column;padding: 10px 50px;margin: 10px }
        </style>
        """
st.markdown(css,unsafe_allow_html=True)
    
def hor_row_builder(horizontal_lines,color:int=0):
    """creates the html of the horizontal lines for a row

    Parameters
    ----------
    horizontal_lines : ndarray
        line states of the respective row of horizontal lines
    color : int,list, optional
        color to apply to the html objects, by default 0

    Returns
    -------
    html
        html object of the board row
    """
    if type(color) is list:
        color1,color2,color3 = color
    else:
        color1,color2,color3 = color,color,color

    dot = div(_class="dot")
    lines = [div(_class="horizContainer",style=styles(background_color=rgb(255,255,255) if i else rgb(color1,color2,color3))) for i in horizontal_lines]
    l = []
    for i in lines:
        l.append(dot)
        l.append(i)
    l.append(dot)

    row = div(_class="row")(l)
    return row

def ver_row_builder(vertical_lines,owners,values,color:int=0):
    """creates the html of the vertical lines for a row

    Parameters
    ----------
    vertical_lines : ndarray
        line states of the respective row of vertical lines
    owners : ndarray
        2d array of the box owners
    values : ndarray
        2d array of the box values
    color : int,list, optional
        color to apply to the html objects, by default 0

    Returns
    -------
    html
        html object of the board row
    """
    if type(color) is list:
        color1,color2,color3 = color
    else:
        color1,color2,color3 = color,color,color

    lines = [div(_class="vertContainer",style=styles(background_color=rgb(255,255,255) if i else rgb(color1,color2,color3))) for i in vertical_lines]
    boxes = [div(_class="box",style=styles(
        background_color=rgba(255,0,0,0.5) if o == 1 else (rgba(0,255,0,0.5) if o == 2 else rgb(255,255,255))
    ))(v) for o,v in zip(owners,values)]

    l = [j for i in zip(lines,boxes) for j in i]
    l.append(lines[-1])

    row = div(_class="row")(l)
    return row

def board_builder(hor_state,ver_state,values,owners,color=0):
    """function to generate the html of the board

    Parameters
    ----------
    hor_state : ndarray
        line states of the respective row of horizontal lines
    ver_state : ndarray
        line states of the respective row of vertical lines
    values : ndarray
        2d array of the box values
    owners : ndarray
        2d array of the box owners
    color : int,list, optional
        color to apply to the html objects, by default 0

    Returns
    -------
    html
        html object of the board
    """
    
    global size

    ver_state = ver_state.T

    rows = []
    for i in range(size):
        #hor_lines = hor_state[i*size:(i+1)*size]
        hor_lines = hor_state[i]
        #ver_lines = ver_state[i*(size+1):(i+1)*(size+1)]
        ver_lines = ver_state[i]
        #vals = values[i*size:(i+1)*size]
        vals = values[i]
        #own = owners[i*size:(i+1)*size]
        own = owners[i]

        rows.append(hor_row_builder(hor_lines,color))
        rows.append(ver_row_builder(ver_lines,own,vals,color))
    hor_lines = hor_state[-1]
    rows.append(hor_row_builder(hor_lines,color))

    board = div(_id="board")(
        div(_id="game-board")(rows)
    )

    return board

def block_to_line(block_index,direction):
    """Converts the block index to a line index given the direction to look

    Parameters
    ----------
    block_index : int
        block index for flattened 2d board
    direction : str
        direction to look

    Returns
    -------
    int
        the line index of the line bordering the box in the given direction
    """
    global size, num_lines
    i = block_index // size
    j = block_index % size
    
    if direction == "Above":
        return block_index
    elif direction == "Below":
        return block_index + size
    elif direction == "Left":
        return num_lines + j*size + i
    elif direction == "Right":
        return num_lines + (j+1)*size + i
    else:
        return 0

def vertical_line_to_block(vert_line_index):
    """Converts an index from the line array to a 2d block index. The index should correspond to a vertical line.

    Parameters
    ----------
    vert_line_index : int
        an index in the line array

    Returns
    -------
    [list,None]
        a list of 2 indexes or None per block
    """
    global size
    
    i,j = vert_line_index%size, vert_line_index//size
    right = [i,j]
    left = [i,j-1]
    
    blocks = [
        left if vert_line_index >= size else None,
        right if vert_line_index < size**2 else None
    ]
    
    return blocks
    
def horizontal_line_to_block(hori_line_index):
    """Converts an index from the line array to a 2d block index. The index should correspond to a horizontal line

    Parameters
    ----------
    hori_line_index : int
        an index in the line array

    Returns
    -------
    [list,None]
        a list of 2 indexes or None per block
    """
    global size
    
    i,j = hori_line_index//size, hori_line_index%size
    above = [i-1,j]
    below = [i,j]
    
    blocks = [
        above if hori_line_index >= size else None,
        below if hori_line_index < size**2 else None
    ]
    
    return blocks

def available_moves(lines):
    """returns all the indexes where lines are available

    Parameters
    ----------
    lines : ndarray
        array of booleans representing if the line at the index is available

    Returns
    -------
    ndarray
        array of available indexes
    """
    #np.where is easy way to get indexes of trues
    return np.where(lines)[0]

def calculate_score(owners):
    """Calculates the score of the board given the owners

    Parameters
    ----------
    owners : ndarray
        2d array of the box owners

    Returns
    -------
    int
        score of the game
    """
    global values
    
    #new_owners = owners.copy()
    #new_owners[np.argwhere(new_owners==2)] = -1
    
    #use the other functionality of where. getting a bug with argwhere
    new_owners = np.where(owners==2,-1,owners)
    
    #multiply to get indivudual scores from each block
    #print(values*new_owners)
    
    #sum the contributions
    return np.sum(values*new_owners)

def add_line_to_block(line_index,owners,line_count,player):
    """adds a line to the bordering blocks and updates the owners and line count

    Parameters
    ----------
    line_index : int
        index of the line added
    owners : ndarray
        2d array of the box owners
    line_count : ndarray
        2d array showing how many bordering lines a box has
    player : int
        the player identifier

    Returns
    -------
    ndarray
        The updates owners
    ndarray
        The updated line count
    """
    global num_lines
    
    new_owners = owners.copy()
    new_line_count = line_count.copy()
    
    #condition for horizontal line
    if line_index < num_lines:
        blocks = horizontal_line_to_block(line_index)
        
    #vertical line
    else:
        #line_index - num_lines 
        blocks = vertical_line_to_block(line_index-num_lines)
        
    
    #check both blocks. easier this way, less code instead of doing both explicitly
    for i in blocks:
        if i:
            #get the count of lines on the box
            lc = new_line_count[i[0],i[1]]
            #update the line count if not already 4
            if lc < 4:
                #if adding the last line to the block
                if lc == 3:
                    #update the owner to the given player
                    new_owners[i[0],i[1]] = player
                new_line_count[i[0],i[1]] += 1
                
    return new_owners, new_line_count

def minimax_max(lines,owners,line_count,depth,alpha,beta):
    """Performs a minimax MAX step

    Parameters
    ----------
    lines : ndarray
        array of the availability of lines
    owners : ndarray
        2d array of the box owners
    line_count : ndarray
        2d array of the number of lines bordering the box
    depth : int
        the current remaining depth before termination
    alpha : int
        the minimax alpha value
    beta : int
        the minimax beta value

    Returns
    -------
    int
        The final score of at termination
    [int,None]
        The move to take or None if no good moves
    """
    global values
    
    moves = available_moves(lines)
    
    #negative infinity
    v = -inf
    
    #reached bottom of the tree
    if depth <= 0 or moves.size == 0:
        #calculate score using owners and return
        score = calculate_score(owners)
        #have to return 2nd item to prevent unpack error
        return score,None
    elif moves.size > 0:
        #print("MAX", depth, f"available moves {moves}")
        
        selected_move = None
        #perform all moves
        #do pruning
        for m in moves:
            #print("MAX", depth, f"performing move {m}")
            
            #create a copy to prevent accidental mutation
            new_move = lines.copy()
            #set the move as taken
            new_move[m] = False

            #update line count and owner
            new_owners, new_line_count = add_line_to_block(m,owners,line_count,1)

            #perform min step
            #ignore the returned move, if any
            m_min,_ = minimax_min(new_move,new_owners,new_line_count,depth-1,alpha,beta)
            #update v
            #v = max(v,m_min)
            if v < m_min:
                v = m_min
                selected_move = m

            if v >= beta:
                #print("MAX", "early stop", f"v: {v}", f"beta: {beta}")
                #early exit step in pruning
                return v,m
            
            #alpha tracking
            alpha = max(alpha,v)
            
        return v,selected_move
    #not reached max depth and no moves available
    else:
        #calculate score using owners and return
        score = calculate_score(owners)
        #have to return 2nd item to prevent unpack error
        return score,None

def minimax_min(lines,owners,line_count,depth,alpha,beta):
    """Performs a minimax MIN step

    Parameters
    ----------
    lines : ndarray
        array of the availability of lines
    owners : ndarray
        2d array of the box owners
    line_count : ndarray
        2d array of the number of lines bordering the box
    depth : int
        the current remaining depth before termination
    alpha : int
        the minimax alpha value
    beta : int
        the minimax beta value

    Returns
    -------
    int
        The final score of at termination
    [int,None]
        The move to take or None if no good moves
    """
    global values
    
    moves = available_moves(lines)
    
    # infinity
    v = inf
    
    #reached bottom of the tree
    if depth <= 0 or moves.size == 0:
        #calculate score using owners and return
        score = calculate_score(owners)
        #have to return 2nd item to prevent unpack error
        return score,None
    elif moves.size > 0:
        #print("MIN", depth, f"available moves {moves}")
        
        selected_move = None

        #perform all moves
        #do pruning
        for m in moves:
            #print("MIN", depth, f"performing move {m}")
            
            #create a copy to prevent accidental mutation
            new_move = lines.copy()
            #set the move as taken
            new_move[m] = False

            #update line count and owner
            new_owners, new_line_count = add_line_to_block(m,owners,line_count,2)

            #perform max step
            #ignore the returned move, if any
            m_max,_ = minimax_max(new_move,new_owners,new_line_count,depth-1,alpha,beta)
            #update v
            #v = min(v,m_max)
            if v > m_max:
                v = m_max
                selected_move = m

            if v <= alpha:
                #print("MIN", "early stop", f"v: {v}",f"alpha: {alpha}")
                #early exit step in pruning
                return v,m

            #beta tracking
            beta = min(beta,v)

        return v,selected_move
    #not reached max depth and no moves available
    else:
        #calculate score using owners and return
        score = calculate_score(owners)
        #have to return 2nd item to prevent unpack error
        return score,None

def alpha_beta(lines,owners,line_count,depth):
    """Gets the move that alpha should make

    Parameters
    ----------
    lines : ndarray
        array of the availability of lines
    owners : ndarray
        2d array of the box owners
    line_count : ndarray
        2d array of the number of lines bordering the box
    depth : int
        the current remaining depth before termination

    Returns
    -------
    int
        The line index of the move to make
    """
    v,move = minimax_max(lines,owners,line_count,depth,-inf,inf)
    #if move:
    #    return move
    #else:
    #   return np.random.choice(available_moves(lines))
    return move

def beta_alpha(lines,owners,line_count,depth):
    """Gets the move that beta should make

    Parameters
    ----------
    lines : ndarray
        array of the availability of lines
    owners : ndarray
        2d array of the box owners
    line_count : ndarray
        2d array of the number of lines bordering the box
    depth : int
        the current remaining depth before termination

    Returns
    -------
    int
        The line index of the move to make
    """
    v,move = minimax_min(lines,owners,line_count,depth,-inf,inf)
    #if move:
    #    return move
    #else:
    #    return np.random.choice(available_moves(lines))
    return move

def show_board(owners,values,lines,col=None):
    """Streamlit visualization function for debugging

    Parameters
    ----------
    owners : ndarray
        the 2d array of the box owners
    values : ndarray
        values of the boxes
    lines : ndarray
        the state of the line availability
    col : st.beta_column, optional
        streamlit column object to write to, by default None
    """
    global size, num_lines

    
    hor_state = lines[:num_lines].reshape((size+1,size))
    ver_state = lines[num_lines:].reshape((size+1,size))
    board = board_builder(hor_state,ver_state,values,owners)
    if col:
        col.markdown(
            str(board),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            str(board),
            unsafe_allow_html=True
        )

def show_progress(owners,values,move,lines,color=0,comment=None,container=None):
    """Streamlit display function to show 2 boards side by side for visualization and debugging

    Parameters
    ----------
    owners : ndarray
        the 2d array of the box owners
    values : ndarray
        values of the boxes
    move : int
        line index of the move taken
    lines : ndarray
        the state of the line availability
    color : int, optional
        the color to apply to the html objects, by default 0
    comment : str, optional
        the string output to explain the action, by default None
    container : st.empty, optional
        streamlit container object for the visualization, by default None
    """
    global size, num_lines, board_index

    l = np.ones(2*num_lines,dtype=np.bool)
    if move >= 0:
        l[move] = False
    o = np.zeros_like(owners)
    

    hor_state = lines[:num_lines].reshape((size+1,size))
    ver_state = lines[num_lines:].reshape((size+1,size))

    move_hor_state = l[:num_lines].reshape((size+1,size))
    move_ver_state = l[num_lines:].reshape((size+1,size))

    if container:
        if comment:
            cols = container.beta_columns(3)
        else:
            cols = container.beta_columns(2)
    else:
        if comment:
            cols = st.beta_columns(3)
        else:
            cols = st.beta_columns(2)


    board = board_builder(hor_state,ver_state,values,owners)
    cols[0].markdown(
        str(board),
        unsafe_allow_html=True
    )

    board_index = np.arange(0,size*size).reshape((size,size))
    board = board_builder(move_hor_state,move_ver_state,board_index,o,color)
    cols[1].markdown(
        str(board),
        unsafe_allow_html=True
    )

    if comment:
        cols[2].write(comment)

def ai_v_ai(lines,owners,line_count):
    """Creates an AI vs AI game

    Parameters
    ----------
    lines : ndarray
        the starting line states
    owners : ndarray
        the 2d array of the box owners
    line_count : ndarray
        the 2d array of the number of lines bordering each box

    Returns
    -------
    [type]
        [description]
    """
    global size, depth

    a = True
    while(available_moves(lines).size > 0):
        if a:
            move = alpha_beta(lines,owners,line_count,depth)
            st.info("ALPHA's move")
            if move:
                comment = f"alpha selected move {move}"
            else:
                move = np.random.choice(available_moves(lines))
                comment = "alpha has no good moves, so it selected a random move"

            lines[move] = False
            owners, line_count = add_line_to_block(move,owners,line_count,1)
            a = not a
            show_progress(owners,values,move,lines,[255,0,0],comment)
        else:

            move = beta_alpha(lines,owners,line_count,depth)
            st.info("BETA's move")
            if move:
                comment = f"beta selected move {move}"
            else:
                move = np.random.choice(available_moves(lines))
                comment = "beta has no good moves, so it selected a random move"


            #print(f"available moves {available_moves(lines)}")
            #move = int(input("pick index"))
            
            lines[move] = False
            owners, line_count = add_line_to_block(move,owners,line_count,2)
            a = not a
            show_progress(owners,values,move,lines,[0,255,0],comment)

        #show_board(owners,values,lines)
        #debug_cols(lines,values,owners,line_count)
    final_score = calculate_score(owners)
    if final_score > 0:
        st.error(f"ALPHA wins. final score: {final_score}")
    elif final_score == 0:
        st.warning("Tie game")
    else:
        st.success(f"BETA wins. final score: {-final_score}")
    return owners


def human_vs_ai():
    """Starts a game with the human making the first move.
    The opposing agent is the min of the minimax algorithm.
    """
    global size, depth
    #sess.human = True

    cols = st.beta_columns(3)
    blocks = cols[0].number_input("block number",0,size*size-1)
    direction = cols[1].selectbox("Direction",["Above","Below","Left","Right"])
    submit = st.button("Submit")

    empty = st.empty()
    empty2 = st.empty()
    empty3 = st.empty()

    
    last_comment = ""

    while(available_moves(sess.lines).size > 0):
        if sess.human:
            empty.info(f"Human's move")
            if submit:
                move = block_to_line(blocks,direction)
                sess.move = move
                
                comment = f"Human selected move {move}"
                last_comment = comment

                sess.lines[move] = False
                sess.owners, sess.line_count = add_line_to_block(move,sess.owners,sess.line_count,1)
                sub_cols = empty2.beta_columns(3)
                sub_cols[0].header("Board state")
                sub_cols[1].header("Move made")
                show_progress(sess.owners,sess.values,move,sess.lines,[255,0,0],comment,empty3)
                sess.human = False
                submit = False
            else:
                sub_cols = empty2.beta_columns(3)
                sub_cols[0].header("Board state")
                sub_cols[1].header("Move made")
                show_progress(sess.owners,sess.values,sess.move,sess.lines,[0,255,0],last_comment,empty3)
            
        else:
            empty.info(f"BETA's move")
            sleep(2)

            move = beta_alpha(sess.lines,sess.owners,sess.line_count,depth)
            if move:
                comment = f"beta selected move {move}"
                last_comment = comment
            else:
                move = np.random.choice(available_moves(sess.lines))
                comment = "beta has no good moves, so it selected a random move"
                last_comment = comment
            sess.move = move
            
            sess.lines[move] = False
            sess.owners, sess.line_count = add_line_to_block(move,sess.owners,sess.line_count,2)
            sub_cols = empty2.beta_columns(3)
            sub_cols[0].header("Board state")
            sub_cols[1].header("Move made")
            show_progress(sess.owners,sess.values,move,sess.lines,[0,255,0],comment,empty3)
            sess.human = True
            submit = False
            
    score = calculate_score(sess.owners)
    if score > 0:
        st.error(f"The HUMAN won with score {score}")
    elif score == 0:
        st.warning("Tie game")
    else:
        st.error(f"The AI won with score {-score}")


    
        

game_mode = st.sidebar.selectbox("Game mode",["<Select>","Human vs AI","AI vs AI"],0)

reset = st.sidebar.button("RESET")

if reset:
    #create the random board
    values = np.random.randint(1,6,size=[size,size])
    board_index = np.arange(0,size*size).reshape((size,size))

    owners = np.zeros_like(values)
    line_count = np.zeros_like(values)

    #true means a line is available
    lines = np.ones(2*num_lines,dtype=np.bool)

    sess.values = values
    sess.owners = owners
    sess.line_count = line_count
    sess.lines = lines

if game_mode == "<Select>":
    st.title("Please select a game mode")
elif game_mode == "Human vs AI":
    st.title("Human vs AI")
    human_vs_ai()
elif game_mode == "AI vs AI":
    st.title("AI vs AI")
    ai_v_ai(lines,owners,line_count)
