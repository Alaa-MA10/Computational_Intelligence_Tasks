from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
import Neural_Network as NN


mainWin = Tk()
mainWin.title('Learning Program')
mainWin.geometry('700x300')

image = ImageTk.PhotoImage(Image.open("Neural-Architecture.png"))
image_panel = Label(mainWin, image=image)
image_panel.pack(side='top', fill='both', expand='yes')
image_panel.image = image

# Learning Rate [Label & TextBox]
Label(mainWin, text="Learning Rate", font=('Bahnschrift', 11)).place(x=400, y=50)
LearningRate_txt = Entry(mainWin)
LearningRate_txt.place(x=510, y=50)

# Epochs [Label & TextBox]
Label(mainWin, text="Epochs", font=("Bahnschrift", 11)).place(x=430, y=100)
epochs_txt = Entry(mainWin)
epochs_txt.place(x=510, y=100)

# Number of hidden layers [Label & TextBox]
Label(mainWin, text="Number of Hidden Layers", font=('Bahnschrift', 11)).place(x=50, y=50)
HiddenLayersNum_txt = Entry(mainWin)
HiddenLayersNum_txt.place(x=230, y=50)

# Number of neurons  [Label & TextBox]
Label(mainWin, text="Number of Neurons", font=("Bahnschrift", 11)).place(x=80, y=100)
Neurons_txt = Entry(mainWin)
Neurons_txt.place(x=230, y=100)

# Select Activation Function
Label(mainWin, text="Select Activation Function", font=('Bahnschrift', 11)).place(x=40, y=150)
selected_activation_val = StringVar()
select_activation_func = ttk.Combobox(mainWin, textvariable=selected_activation_val, width=17)
select_activation_func['values'] = ('Sigmoid', 'Hyperbolic Tangent')
select_activation_func.place(x=230, y=150)

# Check Bias
Bias_Selected = IntVar()
Checkbutton(mainWin, text="Bias", variable=Bias_Selected, padx=10, font=('Arial', 11, 'bold')).place(x=510, y=150)


learning_rate = 0.0
epochs = 0
hidden_layers = 0
neurons_list = []
activation_method = ''


def Run():
    state = check_data()

    if state == 1:
        NN.Run(epochs, learning_rate, hidden_layers, neurons_list, activation_method, Bias_Selected.get())
        evaluation_message = "Confusion Matrix:\n" + str(NN.con_matrix) + "\nAccuracy: " + str(NN.accuracy) + "%"
        messagebox.showinfo("Evaluation Result", evaluation_message)


Button(mainWin, text="RUN", command=Run, width=10, font=('Courier', 12, 'bold'), bg='wheat').place(x=310, y=230)


def check_data():

    if LearningRate_txt.get() == '':
        messagebox.showinfo("Warning", "Please Enter Learning Rate")
        return -1
    if epochs_txt.get() == '':
        messagebox.showinfo("Warning", "Please Enter Epochs number")
        return -1
    if HiddenLayersNum_txt.get() == '':
        messagebox.showinfo("Warning", "Please Enter Hidden Layers number")
        return -1
    if Neurons_txt.get() == '':
        messagebox.showinfo("Warning", "Please Enter Neurons numbers")
        return -1
    if selected_activation_val.get() == '':
        messagebox.showinfo("Warning", "Please Select Activation function")
        return -1

    global learning_rate
    learning_rate = float(LearningRate_txt.get())

    global epochs
    epochs = int(epochs_txt.get())

    global activation_method
    activation_method = selected_activation_val.get()

    global hidden_layers
    hidden_layers = int(HiddenLayersNum_txt.get())

    global neurons_list
    neurons_list = (Neurons_txt.get()).split(',')   # create List[string]
    neurons_list = [int(i) for i in neurons_list]    # convert to List[int]

    if len(neurons_list) != hidden_layers:
        messagebox.showinfo("Warning", "Neurons' numbers not enough for hidden layers")
        return -1

    return 1


mainWin.mainloop()
