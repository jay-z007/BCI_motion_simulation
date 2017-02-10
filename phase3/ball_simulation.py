from Tkinter import *
from random import randint

class Ball:
    def __init__(self, canvas, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.ball = canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill="red")

    def move_ball(self, deltax):
        deltay = 0
        self.canvas.move(self.ball, deltax, deltay)
        print "pos : (",self.x1, ",", self.y1, ")"
