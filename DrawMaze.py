from tkinter import *
from PIL import ImageTk, Image, ImageGrab
import numpy as np
import io
from Tunnel2Goal import *
from ISR import *
from CIT import *
from CMU import *
from TunnelToGoal3 import *
from TunnelToGoal4 import *

class DrawMaze:
    def __init__(self, maze):
        self.root = Tk()
        tiles = []
        for i in range(16):
            tiles.append(ImageTk.PhotoImage(Image.open('MazeDraw/mazetile{0}.jpg'.format(i))))
        maxy = len(maze.map)
        maxx = len(maze.map[0])
        self.canvas = Canvas(self.root, width = 30*maxx+10, height = 30*maxy+10)
        self.canvas.pack()

        for y in range(maxy):
            for x in range(maxx):
                im = maze.map[y][x][1] * 8 + maze.map[y][x][2] * 4 + maze.map[y][x][3] * 2 + maze.map[y][x][4]
                self.canvas.create_image(x * 30+5, y * 30+5, anchor=NW, image=tiles[im])

        self.canvas.update()
        ps = self.canvas.postscript(colormode='color')
        """
        # GhostScriptのインストールが必要
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = ImageTk.PhotoImage(Image.open('CIT.ps'))
        img.save(maze.__class__.__name__+'.png')"""

        """
        # キャプチャもダメ
        print(self.canvas.winfo_rootx(),self.canvas.winfo_rooty())
        print(self.canvas.cget('width'),self.canvas.cget('height'))
        print(self.canvas.winfo_width(),self.canvas.winfo_height())

        wx, wy = self.canvas.winfo_rootx(),self.canvas.winfo_rooty()
        inner_w, inner_h = int(self.canvas.cget('width')),int(self.canvas.cget('height'))
        outer_w, outer_h = self.canvas.winfo_width(),self.canvas.winfo_height()
        ox, oy = (outer_w-inner_w)//2, (outer_h-inner_h)//2
        x0, y0 = 0, 0
        x1, y1 = inner_w, inner_h

        img = ImageGrab.grab((wx+x0+ox,wy+y0+oy,wx+x1+ox,wy+y1+oy))
        img.save(maze.__class__.__name__+'.png')"""

        self.root.mainloop()
        return

if __name__ == '__main__':
    #ret = DrawMaze(Tunnel2Goal())
    #ret = DrawMaze(ISR())
    #ret = DrawMaze(CIT())
    #ret = DrawMaze(CMU())
    #ret = DrawMaze(TunnelToGoal3())
    ret = DrawMaze(TunnelToGoal4())