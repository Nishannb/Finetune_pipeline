Add your own GROQ_API_KEY to run it on your local environment 

Sample prompts are in sample.py file. I seperated it in batches as I was getting api limit error in using groq api for free. 
Prompts are also sythetic prompts created by Chatgpt. 

Reminder for later date: Initially trained with safetensor as true but problem occured when I had to resume the training from checkpoint. 

Also have code for test inference with 2 sample prompt, the dataset is not large enough to have model generate understandable output, but you can still see the difference between first inference call and last inference call after all 6 sample data being trained. 


Main.py is the only file for this project. Run python3 main.py to start training the model. 