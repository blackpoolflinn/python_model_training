import pandas as pd
import tkinter as tk
import sklearn as sk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import subprocess

class modelInstance:

    def __init__(self):
        """Intialise Class Variables"""
        self.df = None
        self.model = None

    # Please add function comment
    def load_dataset():
        """ Selects and loads a chosen data file into a df.
        
        Returns:
            df: df containing the chosen csv or excel file
            
        Raises:
            Error if dataset fails to load
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')
                messagebox.showinfo("Success", "Dataset loaded successfully *but did you check the script!")
                return df
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return None

    # Please add funtion comment
    def train_model(df: pd.DataFrame, features: list, target: str):
        """ Trains model using a given dataset, features and the target variables.
        
        Args:
            df: df containing the data to train the model
            features: Data titles which are to be used to create the prediction
            target: The target variable in which the model has to predict
            
        Returns:
            model: A model trained by the data to predict the target
            
        Raises:
            Exception: Error if the model isn't sucessfully trained
        
        """
        try:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")
        return None

    # Please add function comment
    def make_predictions(model: RandomForestClassifier, df: pd.DataFrame, features: list):
        """Using the given model, uses the given data to give a target prediction.
        
        Args:
            model: sklearn model that has already been trained
            df: pandas df containing data to provide prediction on
            features: data tiles to be used to create the prediction
            
        Returns:
            Visual output of the prediction of the target variable for the data
            
        Raises:
            Exception: Raises exception if prediction is failed to be made
        
        """
        try:
            X_new = df[features]
            predictions = model.predict(X_new)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Predictions:\n{predictions}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make predictions: {e}")

model_instance = modelInstance()

# Please add function comment
root = tk.Tk()
root.title("Student Predictive Grades")

# Please add function comment
load_button = tk.Button(root, text="Load Dataset", command=lambda: model_instance.load_dataset())
load_button.pack(pady=10)

#Please add function comment
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Please add function comment
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Please add function comment
train_button = tk.Button(root, text="Train Model", command=lambda: model_instance.train_model(model_instance.df, features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Please add function comment
predict_button = tk.Button(root, text="Make Predictions", command=lambda: model_instance.make_predictions(model_instance.model, model_instance.df, features_entry.get().split(',')))
predict_button.pack(pady=10)

# Please add function comment
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Please add function comment
root.mainloop()

