import sys
import time
import math
from gameBoard import Board
from marble import Piece
from numpy import zeros, array
from math import sin, log

legal_moves = [[4, 8], [4, 10], [4, 12], [4, 14], [4, 16],
                  [5, 7], [5, 9], [5, 11], [5, 13], [5, 15], [5, 17],
                  [6, 6], [6, 8], [6, 10], [6, 12], [6, 14], [6, 16], [6, 18],
                  [7, 5], [7, 7], [7, 9], [7, 11], [7, 13], [7, 15], [7, 17], [7, 19],
                  [8, 4], [8, 6], [8, 8], [8, 10], [8, 12], [8, 14], [8, 16], [8, 18], [8, 20],
                  [9, 5], [9, 7], [9, 9], [9, 11], [9, 13], [9, 15], [9, 17], [9, 19], [9, 21],
                  [10, 6], [10, 8], [10, 10], [10, 12], [10, 14], [10, 16], [10, 18], [10, 20], [10, 22],
                  [11, 7], [11, 9], [11, 11], [11, 13], [11, 15], [11, 17], [11, 19], [11, 21], [11, 23],
                  [12, 8], [12, 10], [12, 12], [12, 14], [12, 16], [12, 18], [12, 20], [12, 22], [12, 24],
                  [4, 0], [4, 2], [4, 4], [4, 6], [5, 1], [5, 3], [5, 5], [6, 2], [6, 4], [7, 3], [12, 18],
                  [12, 20], [12, 22], [12, 24], [11, 19], [11, 21], [11, 23], [10, 20], [10, 22], [9, 21],
                  [12, 0], [12, 2], [12, 4], [12, 6], [11, 1], [11, 3], [11, 5], [10, 2], [10, 4], [9, 3],
                  [4, 0], [4, 2], [4, 4], [4, 6], [5, 1], [5, 3], [5, 5], [6, 2], [6, 4], [7, 3],[4, 18],
                  [4, 20], [4, 22], [4, 24], [5, 19], [5, 21], [5, 23], [6, 20], [6, 22], [7, 21]]

Player_one = [[0, 12], [1, 11], [1, 13], [2, 10], [2, 12], [2, 14], [3, 9], [3, 11],
                [3, 13], [3, 15]]

Player_two = [[16, 12], [15, 11], [15, 13], [14, 10], [14, 12], [14, 14],
                [13, 9], [13, 11], [13, 13], [13, 15]]

class ChineseCheckers:

    def __init__(node, depth_of_player, rows_size=17, cols_size=25, time_limit=60, c_player=Piece.Piece_RED):
        board = [[[None] * rows_size for __ in range(cols_size)] for _ in range(rows_size)]

        node.Player_one = Player_one
        node.Player_two = Player_two
        node.legal_moves = legal_moves

        for row in range(rows_size):
            for col in range(cols_size):
                if [row, col] in node.Player_one:
                    element = Piece(1, 1, 0, row, col)
                elif [row, col] in node.Player_two:
                    element = Piece(2, 2, 0, row, col)
                elif [row, col] in node.legal_moves:
                    element = Piece(0, 0, 0, row, col)
                else:
                    element = Piece(3, 0, 0, row, col)

                board[row][col] = element

        node.rows_size = rows_size
        node.cols_size = cols_size

        node.time_limit = time_limit
        node.c_player = c_player
        node.board_view = Board(board)
        node.board = board
        node.current_player = Piece.Piece_GREEN
        node.selected_piece = None
        node.valid_moves = []
        node.computing = False
        node.total_plays = 0

        node.depth_of_player = depth_of_player
        node.enable = True

        node.r_goals = [t for row in board
                        for t in row if t.tile == Piece.Tile_RED]
        node.g_goals = [t for row in board
                        for t in row if t.tile == Piece.Tile_GREEN]

        node.board_view.set_status_color("#E50000" if node.current_player == Piece.Piece_RED else "#007F00")

        if node.c_player == node.current_player:
            node.execute_computer_move()

        node.board_view.add_click_handler(node.click_onboard)
        node.board_view.draw_pieces(board=node.board)  # Refresh the board

        print("==============================")
        print("AI opponent enabled:", "no" if node.c_player is None else "yes")
        print("A-B pruning enabled:", "yes" if node.enable else "no")
        print("Turn time limit:", node.time_limit)
        print("Max ply depth:", node.depth_of_player)
        print()

        node.board_view.mainloop()  # Begin tkinter main loop

    ##################################################################################
    def click_onboard(node, row, col):

        if node.computing:  
            return

        new_click = node.board[row][col]

        if new_click.piece == node.current_player:

            node.shape(None)  
            new_click.outline = Piece.Outline_MOVED
            node.v_moves = node.retrieveList(new_click,
                                                  node.current_player)
            node.shape(node.v_moves)

            node.selected_piece = new_click

            node.board_view.draw_pieces(board=node.board)  # Refresh the board

        elif node.selected_piece and new_click in node.v_moves:

            node.shape(None)  
            node.change(node.selected_piece, new_click) 

            node.selected_piece = None
            node.v_moves = []
            node.current_player = (Piece.Piece_RED
                                   if node.current_player == Piece.Piece_GREEN else Piece.Piece_GREEN)

            node.board_view.draw_pieces(board=node.board)  
            winner = node.getWinner()
            if winner:
                node.board_view.set_status("The " + ("green"
                if winner == Piece.Piece_GREEN else "red") + " player has won!")
                node.current_player = None

            elif node.c_player is not None:
                node.execute_computer_move()
        else:
            node.board_view.set_status("Invalid move ")

    ##################################################################################

    def MiniMax(node, deep, maxThePlayer, max_time, maxValue=float("-inf"),
                minValue=float("inf"), maxy=True, Prunes=0, boards=0):

        if deep == 0 or node.getWinner() or time.time() > max_time:
            return node.calculateDistance(maxThePlayer), None, Prunes, boards

        best_move = None
        if maxy:
            best_val = float("-inf")
            Mov = node.secondMoves(maxThePlayer)
        else:
            best_val = float("inf")
            Mov = node.secondMoves((Piece.Piece_RED
                                   if maxThePlayer == Piece.Piece_GREEN else Piece.Piece_GREEN))

        for move in Mov:
            for to in move["2"]:

                if time.time() > max_time:
                    return best_val, best_move, Prunes, boards

                piece = move["1"].piece
                move["1"].piece = Piece.Piece_NONE
                to.piece = piece
                boards += 1

                val, _, new_prunes, new_boards = node.MiniMax(deep - 1,
                                                              maxThePlayer, max_time, maxValue, minValue,
                                                              not maxy, Prunes, boards)
                Prunes = new_prunes
                boards = new_boards

                to.piece = Piece.Piece_NONE
                move["1"].piece = piece

                if maxy and val > best_val:
                    best_val = val
                    best_move = (move["1"].loc, to.loc)
                    maxValue = max(maxValue, val)

                if not maxy and val < best_val:
                    best_val = val
                    best_move = (move["1"].loc, to.loc)
                    minValue = min(minValue, val)

                if node.enable and minValue <= maxValue:
                    return best_val, best_move, Prunes + 1, boards

        return best_val, best_move, Prunes, boards

    ##################################################################################
    def execute_computer_move(root):

        # Print out search information
        current_turn = (root.total_plays // 2) + 1
        print("Turn", current_turn, "Computation")
        print("=================" + ("=" * len(str(current_turn))))
        print("Executing search ...", end=" ")
        sys.stdout.flush()

        root.computing = True
        root.board_view.update()
        max_time = time.time() + root.time_limit

        start = time.time()
        _, move, prunes, boards = root.MiniMax(root.depth_of_player,
                                               root.c_player, max_time)
        end = time.time()

        print("complete")
        print("Time to compute:", round(end - start, 4))
        print("Total boards generated:", boards)
        print("Total prune events:", prunes)

        root.shape(None)  
        move_from = root.board[move[0][0]][move[0][1]]
        move_to = root.board[move[1][0]][move[1][1]]
        root.change(move_from, move_to)

        root.board_view.draw_pieces(board=root.board)  

        winner = root.getWinner()
        if winner:
            root.board_view.set_status("The " + ("green"
                                                 if winner == Piece.Piece_GREEN else "red") + " player has won!")
            root.board_view.set_status_color("#ffffff")
            root.current_player = None
            root.current_player = None

            print()
            print("Final Result")
            print("===========")
            print("The winner:", "green"
            if winner == Piece.Piece_GREEN else "red")
            print("Total Number of plays:", root.total_plays)

        else: 
            root.current_player = (Piece.Piece_RED
                                   if root.current_player == Piece.Piece_GREEN else Piece.Piece_GREEN)

        root.computing = False
        print()
   
    ##################################################################################
    def secondMoves(node,curPlayer=1):
        validMoves = []
        state = 0
        for  i in range(node.cols_size):
            for j in range(node.rows_size):
                state = node.board[j][i]
                if state.piece == curPlayer:   
                   element = {"1": state , "2": node.retrieveList(state, curPlayer)}
                   validMoves.append(element)
                else:
                    continue
        return validMoves

    ##################################################################################
    def retrieveList(node, board, curPlayer, validMoves=[], nerb=True ):
        list = [Piece.Tile_NONE,Piece.Tile_GREEN,Piece.Tile_RED]
        if board.tile != curPlayer:
            list.remove(curPlayer)
        if board.tile != Piece.Tile_NONE and board.tile != curPlayer:
            list.remove(Piece.Tile_NONE)
        if validMoves == []:
            validMoves = []
        for h in [[-1, -1], [-1, 1], [0, 2], [1, 1], [1, -1], [0, -2]]:

            s3,s4 = h
            
            

            if ((board.loc[0] + s3 == board.loc[0]  and board.loc[1] + s4 == board.loc[1]) or
                    board.loc[0] + s3 < 0 or board.loc[1] + s4 < 0 or
                    board.loc[0] + s3 >= node.rows_size or board.loc[1] + s4 >= node.cols_size):
                continue

            
            if (node.board[board.loc[0] + s3][board.loc[1] + s4]).tile not in list:
                continue

            if (node.board[board.loc[0] + s3][board.loc[1] + s4]).piece == Piece.Tile_NONE:
                if nerb:  
                    validMoves.append(node.board[board.loc[0] + s3][board.loc[1] + s4])
                continue



            if (board.loc[0] + s3 + s3 < 0 or board.loc[1] + s4 + s4 < 0 or
                    board.loc[0] + s3 + s3 >= node.rows_size or board.loc[1] + s4 + s4 >= node.cols_size):
                continue

            
            if (node.board[board.loc[0] + s3 + s3][board.loc[1] + s4 + s4]) in validMoves or ((node.board[board.loc[0] + s3 + s3][board.loc[1] + s4 + s4]).tile not in list):
                continue

            if (node.board[board.loc[0] + s3 + s3][board.loc[1] + s4 + s4]).piece == Piece.Piece_NONE:
                validMoves.insert(0, (node.board[board.loc[0] + s3 + s3][board.loc[1] + s4 + s4]))  # Prioritize jumps
                node.retrieveList((node.board[board.loc[0] + s3 + s3][board.loc[1] + s4 + s4]), curPlayer, validMoves, False)

        return validMoves

    ##################################################################################

    def change(node,fState, nState):

        if fState.piece == Piece.Piece_NONE or nState.piece != Piece.Piece_NONE:
            node.board_view.set_status("Invalid move")
            return
        nState.piece = fState.piece
        fState.piece = Piece.Piece_NONE
        nState.outline = Piece.Outline_MOVED
        fState.outline = Piece.Outline_MOVED
        node.total_plays += 1
        node.board_view.set_status_color("#007F00" if
        node.current_player == Piece.Piece_RED else "#E50000")
        fState.tile = 0
        if node.current_player == Piece.Piece_GREEN:
           nState.tile = 1
        else:
            nState.tile = 2
        node.board_view.set_status(("Green player's" if node.current_player == Piece.Piece_RED else "Red player's") + " turn...")
    
 

 ##################################################################################
  
    def getWinner(root):
    
        if all(g.piece == Piece.Piece_GREEN for g in root.r_goals):
            return Piece.Piece_GREEN
        elif all(g.piece == Piece.Piece_RED for g in root.g_goals):
            return Piece.Piece_RED
        else:
            return None
    ##################################################################################

    def shape(root, tiles=[], shape_type=Piece.Outline_SELECT):
    
        if tiles is None:
            tiles = [j for i in root.board for j in i]
            shape_type = Piece.Outline_NONE

        for tile in tiles:
            tile.outline = shape_type
    ##################################################################################
    def calculateDistance(node, Currentplayer):
        def distanceOfPoint(pL1, pL2):
            return math.sqrt((pL2[0] - pL1[0]) ** 2 + (pL2[1] - pL1[1]) ** 2)
            
        result=0
        for col in range(node.cols_size):
            for row in range(node.rows_size):

                current_board = node.board[row][col]

                if current_board.piece == Piece.Piece_GREEN:
                    #Find distance between 2 points
                    distance = [distanceOfPoint(current_board.loc, g.loc)for g in
                                 node.r_goals if g.piece != Piece.Piece_GREEN]
                    result -= max(distance) if len(distance) else -50

                elif current_board.piece == Piece.Piece_RED:
                     #Find distance between 2 points
                    distance = [distanceOfPoint(current_board.loc, g.loc) for g in
                                 node.g_goals if g.piece != Piece.Piece_RED]
                    result += max(distance) if len(distance) else -50

        if Currentplayer == Piece.Piece_RED:
            result *= -1

        return result
    