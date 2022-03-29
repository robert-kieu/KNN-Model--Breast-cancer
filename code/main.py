import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from tkinter import *
import tkinter as tk
from PIL import ImageTk,Image

from tkinter import messagebox



def Preprocessing(df):
    # Mã hóa biến nhị phân
    #     display(df)
    df['diagnosis'] = np.where(df['diagnosis'] == 'M', 1, 0)

    df = df.drop(['Unnamed: 32', 'id'], inplace=True, axis=1)


def hist_chart(col,df):
    plt.hist(df[col], bins=20, edgecolor='black', density=True, color='#33A7D8')
    plt.title(f'Histogram of {col}')
    plt.axvline(df[col].mean(), color='forestgreen', linestyle='dashdot', linewidth=2, label='mean')
    plt.axvline(df[col].median(), color='#F58D4E', linestyle='dashed', linewidth=2, label='median')
    plt.axvline(df[col].mode().iloc[0], color='#ED1B24', linestyle='solid', linewidth=2, label='mode')
    plt.legend()
    plt.show()

def diagnosis_bar(df):
    a = df['diagnosis'].value_counts()
    plt.bar(x = 'Benign', height = (a[0]/len(df)) * 100, color=['b'], edgecolor='black', width=0.5)
    plt.bar(x = 'Malignant', height = (a[1]/len(df)) * 100, color=['r'], edgecolor='black', width=0.5)
    plt.title('Diagnosis')
    plt.ylabel('Percentage of Each Cases (%)')
    plt.yticks([0,20,40,60,80,100])
    plt.show()


def scatter_chart(col1, col2,df):
    a = sns.jointplot(x=col1, y=col2, data=df,
                      kind='reg',
                      line_kws={"color": "r"})
    a.fig.suptitle(f'Relasionship of {col1} and {col2}')
    a.fig.subplots_adjust(top=0.90)

    plt.show()

def corr_heatmap(df):
    Var_Corr = df.corr(method='spearman')
    plt.figure(figsize=(10, 5))
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, cmap='Blues')
    plt.show()

def box_plot(col,df):
    plt.boxplot(df[col], labels=[f'{col}'],
                meanline=True, showmeans=True)
    plt.plot([], [], linestyle='dashed', linewidth=1, color='g', label='mean')
    plt.plot([], [], '-', linewidth=1, color='orange', label='median')
    plt.ylabel('Value')
    plt.title(f'Boxplot of {col}')
    plt.legend()
    plt.show()

def statistic_show(col,df):
    # Return statistic of the argument
    return df[col].describe()


class KNN_Model():

    # Variable:
    # Data is divided into: train_X_df, val_X_df, train_y_sr, val_y_sr.
    # Pipeline for process: full_pipeline.
    # Result of predict: predict.

    def __init__(self, df):
        self.y_sr = df['diagnosis']  # sr là viết tắt của series
        self.X_df = df.drop('diagnosis', axis=1)

    def split_train_test(self):
        # Tách tập huấn luyện và tập validation theo tỉ lệ 80%:20%
        self.train_X_df, self.val_X_df, self.train_y_sr, self.val_y_sr = \
            train_test_split(self.X_df, self.y_sr,
                             test_size=0.2,
                             stratify=self.y_sr,
                             random_state=0)

    def make_pipeline(self):
        # Make pipe cho 1 progress
        self.full_pipeline = make_pipeline(StandardScaler(),
                                           KNeighborsClassifier(n_neighbors=20))

    def fit_data(self):
        self.full_pipeline.fit(self.train_X_df, self.train_y_sr)

    def pred(self, lst_data):
        self.predict = self.full_pipeline.predict(lst_data)

    def classifi_report(self):
        print("Classification Report is:\n", classification_report(self.val_y_sr, self.predict))
        print("Training Score:\n", self.full_pipeline.score(self.train_X_df, self.train_y_sr) * 100)
        print("Valid Score:\n", self.full_pipeline.score(self.val_X_df, self.val_y_sr) * 100)

class Homepage(tk.Frame):
    def __init__(self, parent, controller):


        tk.Frame.__init__(self, parent)
        self.my_image = ImageTk.PhotoImage(Image.open("homepage.png"))
        label_image = Label(self,image=self.my_image)
        label_image.place(x=180,y=100)
        title = tk.Label(self, text='Explore and predict your data now!!', font=("Arial", 30)).place(x=150, y=50)

        Button = tk.Button(self, text="Statistics description", font=("Arial", 30), command=lambda: controller.show_frame(FirstPage))
        Button.place(x=60, y=400)

        Button = tk.Button(self, text="Visualization", font=("Arial", 30), command=lambda: controller.show_frame(SecondPage))
        Button.place(x=375, y=400)

        Button = tk.Button(self, text="Predict", font=("Arial", 30), command=lambda: controller.show_frame(ThirdPage))
        Button.place(x=585, y=400)

class FirstPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # self.configure(bg='blue')

        Button = tk.Button(self, text="Homepage", font=("Arial", 30),
                           command=lambda: controller.show_frame(Homepage))
        Button.place(x=300, y=400)
        label = tk.Label(self, text='Description of statistic', font=("Arial", 15)).place(x=320, y=0)
        options = ['radius_mean', 'texture_mean', 'perimeter_mean',
                   'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                   'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                   'fractal_dimension_se', 'radius_worst', 'texture_worst',
                   'perimeter_worst', 'area_worst', 'smoothness_worst',
                   'compactness_worst', 'concavity_worst', 'concave points_worst',
                   'symmetry_worst', 'fractal_dimension_worst']
        # statistic_show
        clicked5 = tk.StringVar()
        clicked5.set(options[0])

        drop5 = tk.OptionMenu(self, clicked5, *options)
        drop5.place(x=320, y=40)
        label0 = tk.Label(self, text="count:")
        label0.place(x=320, y=150)
        value0 = tk.Label(self, text='')
        value0.place(x=410, y=150)

        label1 = tk.Label(self, text="min:")
        label1.place(x=320, y=175)
        value1 = tk.Label(self, text='')
        value1.place(x=410, y=175)

        label2 = tk.Label(self, text="mean:")
        label2.place(x=320, y=200)
        value2 = tk.Label(self, text='')
        value2.place(x=410, y=200)

        label3 = tk.Label(self, text="std:")
        label3.place(x=320, y=225)
        value3 = tk.Label(self, text='')
        value3.place(x=410, y=225)

        label4 = tk.Label(self, text="25%:")
        label4.place(x=320, y=250)
        value4 = tk.Label(self, text='')
        value4.place(x=410, y=250)

        label5 = tk.Label(self, text="50%:")
        label5.place(x=320, y=275)
        value5 = tk.Label(self, text='')
        value5.place(x=410, y=275)

        label6 = tk.Label(self, text="75%:")
        label6.place(x=320, y=300)
        value6 = tk.Label(self, text='')
        value6.place(x=410, y=300)

        label7 = tk.Label(self, text="max:")
        label7.place(x=320, y=325)
        value7 = tk.Label(self, text='')
        value7.place(x=410, y=325)
        def show():
            statistic = statistic_show(clicked5.get(),df)
            value0.configure(text=np.round(statistic[0], 5))
            value1.configure( text=np.round(statistic[1], 5))
            value2.configure( text=np.round(statistic[2], 5))
            value3.configure( text=np.round(statistic[3], 5))
            value4.configure( text=np.round(statistic[4], 5))
            value5.configure( text=np.round(statistic[5], 5))
            value6.configure( text=np.round(statistic[6], 5))
            value7.configure( text=np.round(statistic[7], 5))
            return
        statistic_button = tk.Button(self, command=show, text='Display')
        statistic_button.place(x=350, y=80)

class SecondPage(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        canvas = Canvas(self, width=500, height=300)
        canvas.pack()

        # Add a line in canvas widget
        canvas.create_line(5, 110, 1000, 110, fill="grey", width=2)
        # self.configure(bg='green')

        Button = tk.Button(self, text="Homepage", font=("Arial", 30), command=lambda: controller.show_frame(Homepage))
        Button.place(x=300, y=370)

        options = ['radius_mean', 'texture_mean', 'perimeter_mean',
                   'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                   'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                   'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                   'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                   'fractal_dimension_se', 'radius_worst', 'texture_worst',
                   'perimeter_worst', 'area_worst', 'smoothness_worst',
                   'compactness_worst', 'concavity_worst', 'concave points_worst',
                   'symmetry_worst', 'fractal_dimension_worst']
        # Histogram
        label_histogram = tk.Label(self, text="Graph for 1 feature", font=('Arial',20)).place(x=300, y=10)
        clicked1 = tk.StringVar()
        clicked1.set('Histogram')

        drop1 = tk.OptionMenu(self, clicked1, *options)
        drop1.place(x=260, y=50)

        hist_button = tk.Button(master=self,
                                command=lambda: hist_chart(clicked1.get(),df),
                                text="Display")
        hist_button.place(x=280, y=75)

        # Boxplot
        clicked2 = tk.StringVar()
        clicked2.set('Boxplot')

        drop2 = tk.OptionMenu(self, clicked2, *options)
        drop2.place(x=430, y=50)

        box_button = tk.Button(master=self,
                               command=lambda: box_plot(clicked2.get(),df),
                               text="Display")
        box_button.place(x=430, y=75)
        # Scatterplot
        label_scatter = tk.Label(self, text="Scatter plot for 2 features", font=('Arial',20)).place(x=280, y=120)
        # col1
        clicked3 = tk.StringVar()
        clicked3.set('Scatter_col1')
        #
        drop3 = tk.OptionMenu(self, clicked3, *options)
        drop3.place(x=270, y=160)
        # # col2
        clicked4 = tk.StringVar()
        clicked4.set('Scatter_col2')
        #
        drop4 = tk.OptionMenu(self, clicked4, *options)
        drop4.place(x=400, y=160)
        scatter_button = tk.Button(master=self,
                                   command=lambda: scatter_chart(clicked3.get(), clicked4.get(),df),
                                   text="Display")
        scatter_button.place(x=354, y=185)
        #
        canvas.create_line(5, 220, 1000, 220, fill="grey", width=2)
        # # heatmap
        label_heatmap = tk.Label(self, text="Heatmap for all attributes", font=('Arial', 20)).place(x=280, y=230)
        heatmap_button = tk.Button(master=self,
                                   command=lambda: corr_heatmap(df),
                                   text="Heatmap")
        heatmap_button.place(x=348, y=260)
        canvas.create_line(5, 300, 1000, 300, fill="grey", width=2)
        # Diagnosis
        label_diagnosis = tk.Label(self, text="Bar chart for diagnosis column", font=('Arial', 20)).place(x=250, y=310)
        diagnosis_button = tk.Button(master=self,
                                     command=lambda: diagnosis_bar(df),
                                     text="Diagnosis")
        diagnosis_button.place(x=345, y=340)


class ThirdPage(tk.Frame):
    def __init__(self, parent, controller,knn):


        tk.Frame.__init__(self, parent)
        # self.configure(bg='Tomato')

        Button = tk.Button(self, text="Homepage", font=("Arial", 30), command=lambda: controller.show_frame(Homepage))
        Button.grid(row=17, column=30)

        # radius
        # radius_mean
        label_radius_mean = tk.Label(self, text="Radius(mean):")
        label_radius_mean.grid(row=1, column=0)
        entry_radius_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_radius_mean.grid(row=1, column=1)
        # texture_mean
        label_texture_mean = tk.Label(self, text="Texture(mean):")
        label_texture_mean.grid(row=2, column=0)
        entry_texture_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_texture_mean.grid(row=2, column=1)
        # perimeter_mean
        label_perimeter_mean = tk.Label(self, text="Perimeter(mean):")
        label_perimeter_mean.grid(row=3, column=0)
        entry_perimeter_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_perimeter_mean.grid(row=3, column=1)
        # area_mean
        label_area_mean = tk.Label(self, text="Area(mean):")
        label_area_mean.grid(row=4, column=0)
        entry_area_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_area_mean.grid(row=4, column=1)
        # smoothness_mean
        label_smoothness_mean = tk.Label(self, text="Smoothness(mean):")
        label_smoothness_mean.grid(row=5, column=0)
        entry_smoothness_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_smoothness_mean.grid(row=5, column=1)
        # compactness_mean
        label_compactness_mean = tk.Label(self, text="Compactness(mean):")
        label_compactness_mean.grid(row=6, column=0)
        entry_compactness_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_compactness_mean.grid(row=6, column=1)
        # concavity_mean
        label_concavity_mean = tk.Label(self, text="Concavity(mean):")
        label_concavity_mean.grid(row=7, column=0)
        entry_concavity_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_concavity_mean.grid(row=7, column=1)
        # concave points_mean
        label_concave_points_mean = tk.Label(self, text="Concave points(mean):")
        label_concave_points_mean.grid(row=8, column=0)
        entry_concave_points_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_concave_points_mean.grid(row=8, column=1)
        # symmetry_mean
        label_symmetry_mean = tk.Label(self, text="Symmetry(mean):")
        label_symmetry_mean.grid(row=9, column=0)
        entry_symmetry_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_symmetry_mean.grid(row=9, column=1)
        # fractal dimension_mean
        label_fractal_dimension_mean = tk.Label(self, text="Fractal dimension(mean):")
        label_fractal_dimension_mean.grid(row=10, column=0)
        entry_fractal_dimension_mean = tk.Entry(self, width=10, cursor="xterm")
        entry_fractal_dimension_mean.grid(row=10, column=1)

        # radius_se
        label_radius_se = tk.Label(self, text="Radius(se):")
        label_radius_se.grid(row=11, column=0)
        entry_radius_se = tk.Entry(self, width=10, cursor="xterm")
        entry_radius_se.grid(row=11, column=1)
        # texture_se
        label_texture_se = tk.Label(self, text="Texture(se):")
        label_texture_se.grid(row=12, column=0)
        entry_texture_se = tk.Entry(self, width=10, cursor="xterm")
        entry_texture_se.grid(row=12, column=1)
        # perimeter_se
        label_perimeter_se = tk.Label(self, text="Perimeter(se):")
        label_perimeter_se.grid(row=13, column=0)
        entry_perimeter_se = tk.Entry(self, width=10, cursor="xterm")
        entry_perimeter_se.grid(row=13, column=1)
        # area_se
        label_area_se = tk.Label(self, text="Area(se):")
        label_area_se.grid(row=14, column=0)
        entry_area_se = tk.Entry(self, width=10, cursor="xterm")
        entry_area_se.grid(row=14, column=1)
        # smoothness_se
        label_smoothness_se = tk.Label(self, text="Smoothness(se):")
        label_smoothness_se.grid(row=15, column=0)
        entry_smoothness_se = tk.Entry(self, width=10, cursor="xterm")
        entry_smoothness_se.grid(row=15, column=1)
        # compactness_se
        label_compactness_se = tk.Label(self, text="Compactness(se):")
        label_compactness_se.grid(row=1, column=50)
        entry_compactness_se = tk.Entry(self, width=10, cursor="xterm")
        entry_compactness_se.grid(row=1, column=51)
        # concavity_se
        label_concavity_se = tk.Label(self, text="Concavity(se):")
        label_concavity_se.grid(row=2, column=50)
        entry_concavity_se = tk.Entry(self, width=10, cursor="xterm")
        entry_concavity_se.grid(row=2, column=51)
        # concave points_se
        label_concave_points_se = tk.Label(self, text="Concave points(se):")
        label_concave_points_se.grid(row=3, column=50)
        entry_concave_points_se = tk.Entry(self, width=10, cursor="xterm")
        entry_concave_points_se.grid(row=3, column=51)
        # symmetry_se
        label_symmetry_se = tk.Label(self, text="Symmetry(se):")
        label_symmetry_se.grid(row=4, column=50)
        entry_symmetry_se = tk.Entry(self, width=10, cursor="xterm")
        entry_symmetry_se.grid(row=4, column=51)
        # fractal dimension_se
        label_fractal_dimension_se = tk.Label(self, text="Fractal dimension(se):")
        label_fractal_dimension_se.grid(row=5, column=50)
        entry_fractal_dimension_se = tk.Entry(self, width=10, cursor="xterm")
        entry_fractal_dimension_se.grid(row=5, column=51)
        # radius_worst
        label_radius_worst = tk.Label(self, text="Radius(worst):")
        label_radius_worst.grid(row=6, column=50)
        entry_radius_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_radius_worst.grid(row=6, column=51)
        # texture
        # texture_worst
        label_texture_worst = tk.Label(self, text="Texture(worst):")
        label_texture_worst.grid(row=7, column=50)
        entry_texture_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_texture_worst.grid(row=7, column=51)
        # perimeter
        # perimeter_worst
        label_perimeter_worst = tk.Label(self, text="Perimeter(worst):")
        label_perimeter_worst.grid(row=8, column=50)
        entry_perimeter_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_perimeter_worst.grid(row=8, column=51)
        # area
        # area_worst
        label_area_worst = tk.Label(self, text="Area(worst):")
        label_area_worst.grid(row=9, column=50)
        entry_area_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_area_worst.grid(row=9, column=51)
        # smoothness
        # smoothness_worst
        label_smoothness_worst = tk.Label(self, text="Smoothness(worst):")
        label_smoothness_worst.grid(row=10, column=50)
        entry_smoothness_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_smoothness_worst.grid(row=10, column=51)
        # compactness
        # compactness_worst
        label_compactness_worst = tk.Label(self, text="Compactness(worst):")
        label_compactness_worst.grid(row=11, column=50)
        entry_compactness_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_compactness_worst.grid(row=11, column=51)
        # concavity
        # concavity_worst
        label_concavity_worst = tk.Label(self, text="Concavity(worst):")
        label_concavity_worst.grid(row=12, column=50)
        entry_concavity_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_concavity_worst.grid(row=12, column=51)
        # concave points
        # concave points_worst
        label_concave_points_worst = tk.Label(self, text="Concave points(worst):")
        label_concave_points_worst.grid(row=13, column=50)
        entry_concave_points_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_concave_points_worst.grid(row=13, column=51)
        # symmetry
        # symmetry_worst
        label_symmetry_worst = tk.Label(self, text="Symmetry(worst):")
        label_symmetry_worst.grid(row=14, column=50)
        entry_symmetry_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_symmetry_worst.grid(row=14, column=51)
        # fractal dimension
        # fractal dimension_worst
        label_fractal_dimension_worst = tk.Label(self, text="Fractal dimension(worst):")
        label_fractal_dimension_worst.grid(row=15, column=50)
        entry_fractal_dimension_worst = tk.Entry(self, width=10, cursor="xterm")
        entry_fractal_dimension_worst.grid(row=15, column=51)

        def enter_click(event):
            # get data
            try:
                data = np.array([float(entry_radius_mean.get()),
                                 float(entry_texture_mean.get()),
                                 float(entry_perimeter_mean.get()),
                                 float(entry_area_mean.get()),
                                 float(entry_smoothness_mean.get()),
                                 float(entry_compactness_mean.get()),
                                 float(entry_concavity_mean.get()),
                                 float(entry_concave_points_mean.get()),
                                 float(entry_symmetry_mean.get()),
                                 float(entry_fractal_dimension_mean.get()),
                                 float(entry_radius_se.get()),
                                 float(entry_texture_se.get()),
                                 float(entry_perimeter_se.get()),
                                 float(entry_area_se.get()),
                                 float(entry_smoothness_se.get()),
                                 float(entry_compactness_se.get()),
                                 float(entry_concavity_se.get()),
                                 float(entry_concave_points_se.get()),
                                 float(entry_symmetry_se.get()),
                                 float(entry_fractal_dimension_se.get()),
                                 float(entry_radius_worst.get()),
                                 float(entry_texture_worst.get()),
                                 float(entry_perimeter_worst.get()),
                                 float(entry_area_worst.get()),
                                 float(entry_smoothness_worst.get()),
                                 float(entry_compactness_worst.get()),
                                 float(entry_concavity_worst.get()),
                                 float(entry_concave_points_worst.get()),
                                 float(entry_symmetry_worst.get()),
                                 float(entry_fractal_dimension_worst.get())])
                knn.pred(data.reshape(1, -1))
                if knn.predict == 1:
                    messagebox.showinfo('Response', 'Malignant')
                else:
                    messagebox.showinfo('Response', 'Benign')
            except:
                messagebox.showinfo('Alert', 'Please fill all blank')

        enter_button = tk.Button(self, text="Enter")
        enter_button.grid(row=16, column=30)
        enter_button.bind("<Button-1>", enter_click)
        enter_button.bind("<Return>", enter_click)

class Application(tk.Tk):
    def __init__(self,knn, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        # creating a window
        window = tk.Frame(self)
        window.pack()

        window.grid_rowconfigure(0, minsize=500)
        window.grid_columnconfigure(0, minsize=765)

        self.frames = {}
        for F in (Homepage,FirstPage, SecondPage,ThirdPage):
            if F == ThirdPage:
                frame = F(window, self,knn)
            else:
                frame = F(window, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Homepage)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()
        self.title("Application")

if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    Preprocessing(df)
    knn = KNN_Model(df)

    knn.split_train_test()

    knn.make_pipeline()

    knn.fit_data()
    app = Application(knn)
    # app.maxsize(780, 500)
    app.mainloop()