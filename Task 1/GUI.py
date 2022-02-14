from tkinter import *
from tkinter import messagebox
import Neural_Network as NN

mainWin = Tk()
mainWin.title('Learning Program')
mainWin.geometry('450x500')


########## Feature List ################

selectFeatureFrame = LabelFrame(mainWin, text="Select Feature", padx=20, pady=20, font='Bahnschrift', width=200)
selectFeatureFrame.place(x=80, y=30)

Feature_List = []
for i in range(4):
    X_selected = IntVar()
    Checkbutton(selectFeatureFrame, text="X" + str((i + 1)), variable=X_selected, font=('Arial', 10)).grid(row=0,
                                                                                                           column=i)
    Feature_List.append(X_selected)

########## Feature List END ################


########## Class List  ################

selectClassFrame = LabelFrame(mainWin, text="Select Class", padx=10, pady=20, font='Bahnschrift', width=150)
selectClassFrame.place(x=20, y=150)

Classes_List = []
for i in range(3):
    C_selected = IntVar()
    Checkbutton(selectClassFrame, text=NN.Iris_data.iloc[i * 50, 4], variable=C_selected,
                font=('Calibri', 12), padx=3).grid(row=0, column=i)
    Classes_List.append(C_selected)

########## Class List END ################

# Learning Rate [Label & TextBox]
Label(mainWin, text="Learning Rate", font=('Bahnschrift', 11)).place(x=30, y=270)
LearningRate_txt = Entry(mainWin)
LearningRate_txt.place(x=155, y=270)

# Epochs [Label & TextBox]
Label(mainWin, text="Epochs", font=("Bahnschrift", 11)).place(x=80, y=320)
epochs_txt = Entry(mainWin)
epochs_txt.place(x=155, y=320)

Bias_Selected = IntVar()
Checkbutton(mainWin, text="Bias", variable=Bias_Selected, padx=10, font=('Arial', 11, 'bold')).place(x=145, y=360)


def Train():
    n, epochs, learning_rate = check_data()
    if n == 1:
        NN.Learn(get_selected_item(Feature_List), get_selected_item(Classes_List),
                           learning_rate, int(epochs), Bias_Selected.get())
        evaluation_message = "Confusion Matrix:\n" + str(NN.confusion_matrix) + "\nAccuracy: " + str(NN.accuracy) + "%"
        messagebox.showinfo("Evaluation Result", evaluation_message)


Button(mainWin, text="Train", command=Train, width=10, font=('Courier', 12, 'bold'), bg='wheat').place(x=250, y=400)


def get_selected_item(check_list):
    selected_items = []

    for i in range(len(check_list)):
        if check_list[i].get() == 1:
            selected_items.append(i + 1)

    return selected_items


def check_data():
    # check select 2 items from Lists
    if len(get_selected_item(Feature_List)) != 2:
        messagebox.showinfo("Warning", "Please Select just 2 Features")
        return -1
    if len(get_selected_item(Classes_List)) != 2:
        messagebox.showinfo("Warning", "Please Select just 2 Classes")
        return -1
    if LearningRate_txt == '':
        messagebox.showinfo("Warning", "Please Enter Learning Rate")
        return -1
    if epochs_txt == '':
        messagebox.showinfo("Warning", "Please Enter Epochs number")
        return -1
    learning_rate = float(LearningRate_txt.get())
    epochs = float(epochs_txt.get())
    return 1, epochs, learning_rate


mainWin.mainloop()
