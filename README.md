# GPT-textgen-skeleton 🤖

*Shakespeare Text Generation Project using GPT architecture - UCLA ACM AI, Projects, Winter '24*

This project is based on Andrej Karpathy's extremely helpful [video tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6014s).

## Training your model on Kaggle

We recommend using Kaggle for training, because it allows you to access GPU runtime for 12 hours. You can also try Google Colab; however Colab has a 90 min GPU usage limit before kicking you out.

1. Navigate to [Kaggle](https://www.kaggle.com/). Click on "Create" and "New notebook" to create a new notebook.

2. Import your Jupyter Notebook `training.ipynb` from your cloned GitHub repository using File > Import Notebook.

3. Load your input text file (`tiny_shakespeare.txt` or other) into the `/kaggle/input` folder.

4. To use the GPU, click the three dots in the top-right corner and select Accelerator > GPU.

5. Most Python packages are already pre-installed in Kaggle, but you might have to import `tiktoken` when using GPT-2 embeddings:

   ```
   !pip install tiktoken
   ```
   
6. If you want your code to run without keeping the tab open, you can click on "Save version" and commit your code. Make sure to save any outputs (e.g. log files) to the `/kaggle/output`, and you should be able to access them in the future.

7. After training is done, you can download the corresponding `.pt` file containing your model's trained parameters.

## Running the main script locally

After training the text generation model remotely, you can download it and run it locally to generate text.

1. Clone this repository and navigate to the base folder.

2. Create and activate a new Conda environment.

3. Install PyTorch, NumPy, and TikToken.

4. Download the trained text generation model and modify the `checkpoint` variable in `main.py` script to correctly reflect where your model's parameters are saved.

5. Run `main.py`. Modify the `max_new_tokens` parameter to make your generated text shorter or longer. Bear in mind that the higher the number of tokens, the longer it will take to generate a response.
