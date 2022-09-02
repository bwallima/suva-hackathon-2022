import pathlib
from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

class TrafficLight():
    def __init__(self):
        width = 600
        height = 850
        #Create an instance of tkinter frame
        self.win= Tk()
        #Set the geometry
        self.win.geometry(f'{width}x{height}')
        #Define function to update the image

        #Create a canvas and add the image into it
        self.canvas= Canvas(self.win, width=width, height= height)
        self.canvas.pack()

        #Open an Image in a Variable
        imagespath = pathlib.Path(__file__).parent.parent.joinpath("images")
        im1= Image.open(imagespath.joinpath("speed_sign.jpg"))
        im2= Image.open(imagespath.joinpath("smiley_green.png"))
        im3= Image.open(imagespath.joinpath("smiley_red.png"))
        imgsize =  (width, height)
        resized_image1 = im1.resize(imgsize, Image.ANTIALIAS)
        resized_image2 = im2.resize(imgsize, Image.ANTIALIAS)
        resized_image3 = im3.resize(imgsize, Image.ANTIALIAS)

        img1= ImageTk.PhotoImage(resized_image1)
        img2= ImageTk.PhotoImage(resized_image2)
        img3= ImageTk.PhotoImage(resized_image3)

        self.imageList = [img1, img2, img3]
        #Add image to the canvas
        self.image_container = self.canvas.create_image(0,0, anchor="nw",image=img1)

        self.win.update()

    def update_image(self, input):
        self.imageList[input]

        self.canvas.itemconfig(self.image_container,image=self.imageList[input])
        self.win.update()

