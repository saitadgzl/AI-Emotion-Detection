#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tkinter as tk
from tkinter import filedialog
import os

# Pencereyi oluştur
window = tk.Tk()
window.title("Face Emotion Recognition")

# Pencere boyutu
window.geometry("980x640")

# Pencere boyutunu sabitle
window.resizable(False, False)

# Pencere ikonu
window.iconbitmap('icon.ico')

#arkaplan rengi
window.configure(bg='orange')

# Create a function that will be called when the first button is clicked
def button1_clicked():
    os.system("video_file.py")
  

 # Load the normal and active background images for the first button
button1_bg_normal = tk.PhotoImage(file='button1_bg.png')
    
# Create a function that will be called when the second button is clicked
def button2_clicked():
    os.system("webcam.py")

# Load the background image for the second button
button2_bg_normal = tk.PhotoImage(file='button2_bg.png')

# Create the first button
button1 = tk.Button(window, text=" ", command=button1_clicked, bg='orange', font=("Arial", 12), image=button1_bg_normal, compound='center', activebackground='yellow', width=300, relief='groove', borderwidth=5)

# Create the second button
button2 = tk.Button(window, text=" ", command=button2_clicked, bg='orange', font=("Arial", 12), image=button2_bg_normal, compound='center', activebackground='yellow', width=300, relief='groove', borderwidth=5)


# Place the buttons in the middle of the window with a gap between them and add padding
button1.pack(side='left', anchor='center', expand=True, fill='both', padx=20, pady=20)
button2.pack(side='left', anchor='center', expand=True, fill='both', padx=20, pady=20)

# Run the main loop
window.mainloop()

