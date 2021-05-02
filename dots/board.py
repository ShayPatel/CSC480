import streamlit as st
from htbuilder import div, styles
from htbuilder.funcs import rgba, rgb

from random import randint


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
    
def hor_row_builder(row_state:str):
    dot = div(_class="dot")
    lines = [div(_class="horizContainer",style=styles(background_color=rgb(255,255,255) if i =="0" else rgb(0,0,0))) for i in row_state]
    l = []
    for i in lines:
        l.append(dot)
        l.append(i)
    l.append(dot)

    row = div(_class="row")(l)
    return row

def ver_row_builder(row_state:str,values:str,player:str):
    lines = [div(_class="vertContainer",style=styles(background_color=rgb(255,255,255) if i =="0" else rgb(0,0,0))) for i in row_state]
    boxes = [div(_class="box",style=styles(
        background_color=rgba(255,0,0,0.5) if p == "1" else (rgba(0,255,0,0.5) if p == "2" else rgb(255,255,255))
    ))(v) for p,v in zip(player,values)]

    l = [j for i in zip(lines,boxes) for j in i]
    l.append(lines[-1])

    row = div(_class="row")(l)
    return row

def board_builder(size:int,hor_state:str,ver_state:str,values:str,player:str):
    rows = []
    for i in range(size):
        hor_lines = hor_state[i*size:(i+1)*size]
        ver_lines = ver_state[i*(size+1):(i+1)*(size+1)]
        vals = values[i*size:(i+1)*size]
        plays = player[i*size:(i+1)*size]

        rows.append(hor_row_builder(hor_lines))
        rows.append(ver_row_builder(ver_lines,vals,plays))
    hor_lines = hor_state[size*size:(size+1)*size]
    rows.append(hor_row_builder(hor_lines))

    board = div(_id="board")(
        div(_id="game-board")(rows)
    )

    return board

def random_generator(size):
    players = "0"*(size**2)
    values = "".join([str(randint(1,5)) for i in range(size**2)])
    
    state = "0"*(2*size*(size+1))
    ver_state = state[-(size*(size+1)):]
    hor_state = state[:(size*(size+1))]

    board = board_builder(size,hor_state,ver_state,values,players)
    st.markdown(
        css + str(board),
        unsafe_allow_html=True
    )


def sample_state():
    size = 5
    players = "0211200000000000000000000"
    values = "3543525523244212444445314"
    ver_state = "011111000010000000000000000000"
    hor_state = "011110111100000000000000000000"
    board = board_builder(size,hor_state,ver_state,values,players)
    st.markdown(
        css + str(board),
        unsafe_allow_html=True
    )

#st.markdown(
#    css + test,
#    unsafe_allow_html=True
#)

#size = 2
#hor_state = "000000000"
#ver_state = "000000000"
#values = "1234"
#players = "0000"

#board = board_builder(size,hor_state,ver_state,values,players)
#st.markdown(
#    css + str(board),
#    unsafe_allow_html=True
#)


#size = 5
#random_generator(size)
sample_state()