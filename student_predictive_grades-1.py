import pandas as pd
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

class modelInstance:

    def __init__(self):
        """Intialise Class Variables"""
        self.df = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.clean_df = None

    def set_df(self, df):
        """Df setter"""
        self.df = df

    def set_clean_df(self, df):
        """Clean df setter"""
        self.clean_df = df

    def set_model(self, model: RandomForestRegressor):
        """Model setter"""
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
            
                # Merge the files together into a df
                merged_df = pd.concat(dataframes, ignore_index=True)
                self.set_df(merged_df)
                self.set_clean_df(None)

                # Show the list of filenames in the GUI
                file_display_text.delete(1.0, tk.END)
                file_display_text.insert(tk.END, "Loaded files:\n")
                for path in file_paths:
                    file_display_text.insert(tk.END, f"- {path.split('/')[-1]}\n")

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

        self.set_clean_df(self.df.drop('student_id', axis=1))

        messagebox.showinfo("Success", "Dataset(s) cleaned")
        

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

        # Cap exam_score to a maximum of 100
        df['exam_score'] = df['exam_score'].clip(upper=100)

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
            features: Data titles which are to be used to create the prediction
            target: The target variable in which the model has to predict
            
        Returns:
            model: A model trained by the data to predict the target
            
        Raises:
            Exception: Error if the model isn't sucessfully trained
        
        """

        if self.clean_df is None:
            if self.df is None:
                messagebox.showerror("Error", "No dataset loaded.")
            else:
                messagebox.showerror("Error", "Dataset not cleaned.")
            return

        try:
            x = self.clean_df[features]
            y = self.clean_df[target]

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            MSE = mean_squared_error(y_test, y_pred)
            MAE = mean_absolute_error(y_test, y_pred)
            messagebox.showinfo("Model Trained",
                                 f"Model trained successfully! \n\n"
                                 f"R squared score: {r2:.2f}\n"
                                 f"Mean squared error: {MSE:.2f}\n"
                                 f"Mean absolute error: {MAE:.2f}")
            metric_text.config(text=f"Model Evaluation Metrics:\n"
                         f"R squared score: {r2:.2f}\n"
                         f"Mean Squared Error (MSE): {MSE:.2f}\n"
                         f"Mean Absolute Error (MAE): {MAE:.2f}")
            self.set_model(model)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")


    def make_predictions(self, features: list):
        """Using the given model, uses the given data to give a target prediction.
        
        Args:
            features: variables to be used to make predictions
            
        Returns:
            Visual output of the prediction of the target variable for the data
            
        Raises:
            Exception: Raises exception if prediction is failed to be made
        
        """

        if self.model is None:
            messagebox.showerror("Error", "No model has been trained.")
            return
        
        if self.clean_df is None:
            if self.df is None:
                messagebox.showerror("Error", "No dataset loaded.")
            else:
                messagebox.showerror("Error", "Dataset not cleaned.")
            return
        
        if features[0] == '':
            messagebox.showerror("Error", "No features selected.")
            return

        try:
            X_new = self.clean_df[features]
            predictions = self.model.predict(X_new)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Predictions:\n{np.round(predictions, 2)}")

            df = pd.DataFrame({'exam_score' : np.round(predictions, 2)})
            df.index.name = 'student_number'
            df.to_csv('predictions.csv')

            messagebox.showinfo("Success", "Predictions made, shown in display and saved to 'predictions.csv'.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to make predictions: {e}")

def is_number(val):
    """Float type checking function"""
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

clean_button = tk.Button(root, text="Clean Dataset", command=lambda: model_instance.data_preprocessing())
clean_button.pack(pady=5)

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

# Draws a textbox that displays the metrics for the model
metric_text = tk.Label(root, text="Model Evaluation Metrics:\n", justify='left', anchor='w', bg='lightgrey', width=40, height=4, relief='raised')
metric_text.pack(pady=5)

# Creates a predict button to predict the target variable for a given dataset and group of features
predict_button = tk.Button(root, text="Make Predictions", command=lambda: model_instance.make_predictions(features_entry.get().split(',')))
predict_button.pack(pady=10)

# Draws the resulting predicted data to a text box
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Draws a textbox that displays which datasets have been loaded
file_display_text = tk.Text(root, height=5, width=80)
file_display_text.pack(pady=5)

# tkinter main loop 
root.mainloop()