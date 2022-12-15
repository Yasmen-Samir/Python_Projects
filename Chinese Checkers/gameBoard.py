import tkinter as t
from turtle import color

from pygame import Color


class Board(t.Tk):

    def __init__(root, mainBoard, *args, **kwargs):

        # Initialize parent tk class
        t.Tk.__init__(root, *args, **kwargs)

        # Setting Titlen
        root.title("Chinese Checkers")
        # Setting Background
        root.configure(background="black")
        #root.iconbitmap(default="E:\Chinese Checkers\google_balls.ico")
        root.resizable(False, False)

        # Save tracking variables
        root.pieces = {}
        root.board = mainBoard
        root.rowSize = len(mainBoard)
        root.columnSize = len(mainBoard[0])

        # Create grid canvas
        root.canvas = t.Canvas(root, width=555, height=600, background="black", highlightbackground="black")
        root.canvas.grid(row=1, column=1,
                         columnspan=root.columnSize, rowspan=root.rowSize)

        # Create status label
        root.status = t.Label(root, anchor="c", font=(None, 16),
                              bg="#212F3D", fg="#212F3D", text="Green player's turn")
        root.status.grid(row=root.rowSize + 3, column=0,
                         columnspan=root.columnSize + 3, sticky="ewns")

        # ---------------------------------------------
        # Shape Initializing
        # ---------------------------------------------
        # Game Shape
        root.canvas.bind("<Configure>", root.draw_pieces)

        # Shape Around the game
        root.columnconfigure(0, minsize=50)
        root.rowconfigure(0, minsize=50)
        root.columnconfigure(root.columnSize + 2, minsize=50)
        root.rowconfigure(root.rowSize + 2, minsize=50)
        # root.rowconfigure(root.r_size + 3, minsize=48)

    # Methods
    def add_click_handler(root, func):
        root.click_handler = func

    def set_status(self, text):
        self.status.configure(text=text)

    def set_status_color(self, color):
        self.status.configure(bg=color)



    def draw_pieces(root, event=None, board=None):

        if board is not None:
            root.board = board

        # Delete old rectangles and save properties
        root.canvas.delete("tile")

        # Can Change Width and Height From here
        cell_width = int(root.canvas.winfo_width() / root.columnSize)
        cell_height = int(root.canvas.winfo_height() / root.rowSize)
        border_size = 2

        # Recreate each rectangle
        for column in range(root.columnSize):
            for row in range(root.rowSize):
                boardMarble = root.board[row][column]
                marbleColor, outline_color = boardMarble.get_marble_colors()

                # Calculate pixel positions
                horizontal_point1 = column * cell_width + border_size / 2
                vertical_point1 = row * cell_height + border_size / 2
                horizontal_point2 = (column + 1) * cell_width - border_size / 2
                vertical_point2 = (row + 1) * cell_height - border_size / 2

                # Render tile
                marble = root.canvas.create_oval(horizontal_point1, vertical_point1, horizontal_point2, vertical_point2,
                                                 tags="tile", width=border_size, fill=marbleColor,
                                                 outline=outline_color)
                root.pieces[row, column] = marble

                # if some event happened
                root.canvas.tag_bind(marble, "<1>", lambda event, row=row,
                                                           column=column: root.click_handler(row, column))

        root.update()


