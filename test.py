"""
To do : faire machine à état : si train.txt existe alors on peut faire générer sinon il faut passer par la case train 
Ensuite faire load model
"""
from easygui import *
import sys
import os
from articles_generator import *
from datetime import datetime

msg_7 = """ Your article has been created, you can find it in the same folder as this program under the name you chose\Bye !"""
msg_6 = """PLease give a word to start the generation (any word will work)"""
msg_5 = """Your model has been trained, it can now generate an article"""
msg_4 = """ Please enter the name you want to give to your generated article,\nIf you don't, a default name will be picked """
msg_3 = """Are you sure about what you picked ?"""
msg_2 =""" Before everything you must train the program \n Once it has been trained, you can generate article of the desired length \n If you haven't trained any 
			\nTo train the program you must click on the Train button and chose a file containing a lot of examples (in a .txt file) which will be used by the program to learn how it should generate the text"""
msg_1 = """Welcome to the Articles Generator.\nThis program is working as follows : 
	\n- First you train the program to a specific style of writing \n- Then you can run the generator to create texts based on the style you fed it
	\n\n\nThis program has been made using Tensorflow and easygui tutorials """
title = "Articles generator"
choices_1 = ["Exit", "Train"]
choices_2 = ["No", "Next"]
choices_0 = ["Exit", "Next"]
choices_3 = ["Exit", "Go"]
choices_4 = ["Exit", "Generate"]
choices_5 = ["Exit", "OK"]
def training(file):
	checkpoint_dir = "./training_checkpoints"
	# Name of the checkpoint files
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
	model = model_train(checkpoint_prefix, file)
	return model
	
def mainloop ():
	# A nice welcome message
	if os.path.exists("saved_data") :
		choices_1.append("Create article")
	ret_val = buttonbox(msg_1, choices=choices_0)
	if ret_val is None: # User closed msgbox
		sys.exit(0)
	while 1:
		choice_1 = buttonbox(msg_2, title, choices=choices_1)
		if choice_1 is "Train":
			training_file = fileopenbox()
			choice_2 = buttonbox(msg_3, title, choices= choices_2)
			if choice_2 is "Next":
				training(training_file)	
				choice_4 = buttonbox(msg_5, title, choices= choices_4)
				if choice_4 is "Generate":
					now = datetime.now()
					dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
					default_name = "generated_article" + dt_string +".txt"
					enterbox(msg=msg_4, title=title, default=default_name, strip=True)
				else :
					sys.exit(0)
			else :
				training_file = fileopenbox()
				training(training_file)
			sys.exit(0)
		if choice_1 is "Create article":
			now = datetime.now()
			dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
			default_name = "generated_article" + dt_string +".txt"
			article_name = enterbox(msg=msg_4, title=title, default=default_name, strip=True)
			if article_name is not "":
				start_string = enterbox(msg=msg_6, title=title, default="Rayan", strip=True)
				res = create_article(start_string)
				with open(article_name, 'w') as handle:
					handle.write(res)
				choice_5 = buttonbox(msg_7, title, choices=choices_5)
				sys.exit(0)
		else :
			sys.exit(0)
			
mainloop()