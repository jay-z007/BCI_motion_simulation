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

    def move_ball(self):
        deltax = randint(0,10)*randint(-1, 1)
        deltay = randint(0,10)*randint(-1, 1)
        self.canvas.move(self.ball, deltax, deltay)
        pos = self.canvas.coords(self.ball)
        print [x+25 for x in pos[0:2]]

#initialize root Window and canvas
root = Tk()
root.title("Balls")
root.resizable(False,False)
canvas = Canvas(root, width = 500, height = 500)
canvas.pack()

#create two ball objects and animate them
ball1 = Ball(canvas, 100, 100, 150, 150)
#ball2 = Ball(canvas, 60, 60, 80, 80)

while int(input()) != 0:
    ball1.move_ball()
#ball2.move_ball()

#root.mainloop()