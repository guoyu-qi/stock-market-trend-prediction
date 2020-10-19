## This code contains several types of tkinter 
## windows that can be used for developing the user interface 

from tkinter import *

class Widgets:

	def __init__(self):
		pass

	def askToSelectOption(self, prompt, options, factor):
		root = Tk()
		root.configure(background="blue")
		root.option_add("*Button.Background", "black")
		root.option_add("*Button.Foreground", "red")
		a = len(prompt) * factor[0]
		b = len(options) * factor[1]
		root.geometry("%dx%d" % (a,b))
		root.title(prompt)
		v = IntVar()
		for i, option in enumerate(options):
			Radiobutton(root, text=option, font=("Times", 10, "bold"), 
			 background="blue", highlightbackground="black", 
			 variable=v, value=i, width = a, indicatoron=0).pack(anchor="n")
		Button(text="Submit", command=root.destroy).pack(anchor = "n")
		root.mainloop()
		return options[v.get()]

	

	def userEntry(self, prompt, entry_list, button_text):
		master = Tk()
		master.configure(background="blue")
		master.option_add("*Button.Background", "black")
		master.option_add("*Button.Foreground", "red")
		master.title(prompt)
		master.grid()
		master.entries = []

		for i in range(len(entry_list)):
			label = Label(master, text=entry_list[i], bg="blue", 
				highlightbackground="black", width=19, font=("Times", 10, "bold"))
			label.grid(row=i, column=0)

			entry = Entry(master, bg="blue", width=19, 
				highlightbackground="black", font=("Times", 10, "bold"))
			entry.grid(row=i, column=1)
			master.entries.append(entry)

		self.values = []
		def getValues():
			for entry in master.entries:
				self.values.append(entry.get())
			master.destroy()

		Button(master, text=button_text, command = getValues).grid(row=6, column =1,  pady=4)
		master.mainloop()
		
		return self.values

	def scrollbar(self, prediction_type_prompt, prediction_type_option):
		root = Tk()
		root.title(prediction_type_prompt)
		root.option_add("*Button.Background", "black")
		root.option_add("*Button.Foreground", "red")
		scrollbar = Scrollbar(root, bg="blue", 
				highlightbackground="black")
		scrollbar.pack( side = "right", fill = "y" )

		mylist = Listbox(root, yscrollcommand = scrollbar.set, bg="blue", 
				highlightbackground="black", width=40, font=("Times", 10, "bold") )
		for prediction_type in prediction_type_option:
		   mylist.insert(END, prediction_type)

		
		self.prediction_type = []
		def getValues():
			self.prediction_type.append(mylist.get(ACTIVE))
			root.destroy()

		mylist.pack()
		scrollbar.config( command = mylist.yview )
		Button(text="Submit", command=getValues).pack(anchor = "s")
		mainloop()
		return self.prediction_type[0]