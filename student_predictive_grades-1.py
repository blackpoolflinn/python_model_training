import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

class modelInstance:

    def __init__(self):
        """Intialise Class Variables"""
        self.df = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def set_df(self, df):
        self.df = df

    def set_model(self, model: RandomForestRegressor):
        self.model = model


    def load_dataset(self):
        """ Selects and loads a chosen data file into a df.
        
        Returns:
            df: df containing the chosen csv or excel file
            
        Raises:
            Error if dataset fails to load
        """
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
        if file_paths:
            try:
                dataframes = []
                for path in file_paths:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                    else:
                        df = pd.read_excel(path, engine='openpyxl')
                    dataframes.append(df)
                
                merged_df = pd.concat(dataframes, ignore_index=True)
                self.set_df(merged_df)
                self.data_preprocessing()
                messagebox.showinfo("Success", "Dataset(s) loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
        else:
            self.set_df(None)

    def data_preprocessing(self) -> pd.DataFrame:
        """Top level function for data preprocessing of the given dataset"""
        
        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded.")

        self.remove_empty_rows()
        self.verify_numerical_categories()
        self.encode_string_categories()

        self.set_df(self.df.drop('student_id', axis=1))
        

    def remove_empty_rows(self) -> pd.DataFrame:
        """Uses regex to convert all empty cells into NAN values and drops empty rows"""
        df = self.df.replace(r'^\s*$', np.nan, regex=True)

        df = df.dropna(how='all')

        # Drop rows where ≥80% of fields are missing
        threshold = int(df.shape[1] * 0.8)
        df = df.dropna(thresh=df.shape[1] - threshold)

        self.set_df(df)

    
    def verify_numerical_categories(self):
        """Verifies the datatype of numerical columns in the dataset"""
        df = self.df

        numerical_categories = ['age','study_hours_per_day','social_media_hours','netflix_hours',
                                        'attendance_percentage','sleep_hours','exercise_frequency',
                                        'mental_health_rating','exam_score']

        for column in numerical_categories:
            # Apply mask to find numeric values
            mask = df[column].apply(is_number)

            # Convert numeric values to float
            df.loc[mask, column] = df.loc[mask, column].astype(float)

            # Calculate the median of the column
            median_value = df.loc[mask, column].median()

            # Replace non-numeric values with the median
            df.loc[~mask, column] = median_value
            df[column] = df[column].astype(float)

        self.set_df(df)


    def encode_string_categories(self):
        """Encodes string categories using label encoder to allow for them to be used in model training"""
        df = self.df

        categorical_cols = [
            'gender', 'part_time_job', 'diet_quality',
            'parental_education_level', 'internet_quality', 'extracurricular_participation'
        ]
        
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        for col in categorical_cols:
            df[col] = self.label_encoder.fit_transform(df[col])
        
        self.set_df(df)


    def train_model(self, features: list, target: str):
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

        if self.df is None:
            messagebox.showerror("Error", "No dataset loaded.")
            return

        try:
            x = self.df[features]
            y = self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            r2_percent = r2 * 100
            messagebox.showinfo("Model Trained", f"Model trained successfully! Estimated accuracy: {r2_percent:.2f}%")
            self.set_model(model)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")


    def make_predictions(self, features: list):
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

        if self.model is None:
            messagebox.showerror("Error", "No model has been trained.")
            return

        try:
            X_new = self.df[features]
            predictions = self.model.predict(X_new)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Predictions:\n{predictions}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make predictions: {e}")

def is_number(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

# Intialise model class
model_instance = modelInstance()

# Intialise Tkinter element and set window title
root = tk.Tk()
root.title("Student Predictive Grades")

# Create a load dataset button for the tkinter window
load_button = tk.Button(root, text="Load Dataset", command=lambda: model_instance.load_dataset())
load_button.pack(pady=10)

# Create an input field for features to train the dataset model on
tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root)
features_entry.pack(pady=5)

# Create a target input field for the prediction variable
tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root)
target_entry.pack(pady=5)

# Creates a train button to train the model once the dataset has been selected, features entered and target variable selected
train_button = tk.Button(root, text="Train Model", command=lambda: model_instance.train_model(features_entry.get().split(','), target_entry.get()))
train_button.pack(pady=10)

# Creates a predict button to predict the target variable for a given dataset and group of features
predict_button = tk.Button(root, text="Make Predictions", command=lambda: model_instance.make_predictions(features_entry.get().split(',')))
predict_button.pack(pady=10)

# Draws the resulting predicted data to a text box
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# tkinter main loop 
root.mainloop()