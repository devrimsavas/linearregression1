import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Years_of_Experience': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5],
    'Salary_in_Thousands': [45, 47.5, 52, 56, 61, 66, 69, 72, 78, 82, 86, 91, 94, 97, 100, 104, 108, 112, 117, 120]
}

#we convert data set to np array . It is not necessary for scipy but sklearn needs . 
X = np.array(data['Years_of_Experience'])
y = np.array(data['Salary_in_Thousands'])

# Scipy Regression
slope, intercept, r, p, stderr = stats.linregress(X, y)

# Sklearn Regression
Xs = X.reshape(-1, 1) #reshape array X
model = LinearRegression().fit(Xs, y)
r2_value = model.score(Xs, y)

def predict_salary():
    try:
        hours = float(entry_hours.get())
        salary_scipy = slope * hours + intercept
        salary_sklearn = model.predict(np.array([[hours]]))[0]
        label_result_scipy.config(text=f"Predicted Salary (Scipy): ${salary_scipy:.2f}k")
        label_result_sklearn.config(text=f"Predicted Salary (Sklearn): ${salary_sklearn:.2f}k")
    except ValueError:
        label_result_scipy.config(text="Invalid input")
        label_result_sklearn.config(text="")

# Tkinter 
root = tk.Tk()
root.title("Linear Regression Analysis")
root.config(bg="#D3D3D3")
root.geometry("1500x600")

# Frame for plots
frame_plots = ttk.Frame(root)
frame_plots.place(x=20,y=20)

# Embedding graphic the Scipy plot
fig1 = plt.Figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111)
ax1.scatter(X, y)
ax1.plot(X, slope * X + intercept, color="blue")
ax1.set_title("Experience & Salary (Scipy model)")
canvas1 = FigureCanvasTkAgg(fig1, master=frame_plots)
canvas1.get_tk_widget().pack(side=tk.LEFT)

# Embedding graphic the Sklearn plot
fig2 = plt.Figure(figsize=(5, 5))
ax2 = fig2.add_subplot(111)
ax2.scatter(X, y)
ax2.plot(X, model.predict(Xs), color="red")
ax2.set_title("Experience & Salary (Sklearn model)")
canvas2 = FigureCanvasTkAgg(fig2, master=frame_plots)
canvas2.get_tk_widget().pack(side=tk.LEFT)

# Treeview for raw data
frame_data = ttk.Frame(root)

frame_data.place(x=1070,y=20)
tree = ttk.Treeview(frame_data, columns=('Years_of_Experience', 'Salary_in_Thousands'), show='headings')
tree.heading('Years_of_Experience', text='Years of Experience')
tree.heading('Salary_in_Thousands', text='Salary (in thousands)')
tree.pack()

for x_val, y_val in zip(X, y):
    tree.insert("", "end", values=(x_val, y_val))

# Frame for prediction and r-values
frame_predict = ttk.Frame(root)

frame_predict.place(x=1070,y=300)

# Entry for hours
label_hours = ttk.Label(frame_predict, text="Enter Years of Experience:")
label_hours.pack(pady=5)
entry_hours = ttk.Entry(frame_predict)
entry_hours.pack(pady=5)

# Prediction Button
btn_predict = ttk.Button(frame_predict, text="Predict Salary", command=predict_salary)
btn_predict.pack(pady=10)

# Labels to show predictions
label_result_scipy = ttk.Label(frame_predict, text="Predicted Salary (Scipy):")
label_result_scipy.pack(pady=5)
label_result_sklearn = ttk.Label(frame_predict, text="Predicted Salary (Sklearn):")
label_result_sklearn.pack(pady=5)

# Labels to show r-values
label_r_scipy = ttk.Label(frame_predict, text=f"R value (Scipy): {r:.4f}")
label_r_scipy.pack(pady=5)
label_r2_sklearn = ttk.Label(frame_predict, text=f"R^2 value (Sklearn): {r2_value:.4f}")
label_r2_sklearn.pack(pady=5)

root.mainloop()
