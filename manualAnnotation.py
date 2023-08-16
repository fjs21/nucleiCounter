import cv2 as cv
# import os
import json
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

from commonFunctions import *
from singleCompositeImage import singleCompositeImage
from settings import Settings

settings = Settings()


class Application(tk.Frame):
    def __init__(self, master = None):
        """Setup tk application."""
        super().__init__(master)
        self.o4neg = None
        self.o4pos = None
        self.unclassified = None
        self.name = None
        self.master = master
        self.master.title("Manual annotation of O4 cells")
        self.pack()
        self.create_widgets()

    def showInstructions(self):
        """Show instructions to user for image classification."""
        t = tk.Toplevel(self, width=400, height=200)
        t.geometry("+740+100")
        msg = tk.Label(t, text="For each marked cell: \n" +
                               "     Press 'y' for POSITIVE\n" +
                               "     Press 'n' for NEGATIVE\n" +
                               "Otherwise, press 'q' to QUIT", padx=20, pady=20,
                       font=tkFont.Font(family="Calibri", size=20), justify=tk.LEFT)
        msg.pack()
        self.master.update()

    def start(self, redo):
        """Start annotation routine."""
        self.master.iconify()
        self.showInstructions()
        self.runAnnotation(self.name.get(), redo)
        self.master.destroy()

    def create_widgets(self):
        """Creates widgets on initial window."""
        tk.Label(self,
                 text="""Select experiment to annotate:""",
                 justify=tk.LEFT,
                 padx=20,
                 font=tkFont.Font(family="Calibri", size=12)).pack()

        self.o4pos = tk.IntVar()
        self.o4pos.set(1)
        self.o4neg = tk.IntVar()
        self.o4neg.set(0)
        self.unclassified = tk.IntVar()
        self.unclassified.set(0)

        # determine which cell types will be manually annotated

        self.name = tk.StringVar()
        experiments = list(settings.experiments)
        print(experiments)
        self.name.set(settings.defaults["name"])
        combo = ttk.Combobox(self, values=experiments,
                             width=80, textvariable=self.name,
                             font=tkFont.Font(family="Calibri", size=14))
        combo.pack()

        on = tk.Checkbutton(self,
                            text="O4- cells",
                            variable=self.o4neg,
                            onvalue=1,
                            offvalue=0,
                            font=tkFont.Font(family="Calibri", size=12))
        on.pack()

        op = tk.Checkbutton(self,
                            text="O4+ cells",
                            variable=self.o4pos,
                            onvalue=1,
                            offvalue=0,
                            font=tkFont.Font(family="Calibri", size=12))
        op.pack()

        un = tk.Checkbutton(self,
                            text="unclassified",
                            variable=self.unclassified,
                            onvalue=1,
                            offvalue=0,
                            font=tkFont.Font(family="Calibri", size=12))
        un.pack()

        button = tk.Button(self,
                           text="OK",
                           command=lambda: self.start(False),
                           font=tkFont.Font(family="Calibri", size=12))
        button.pack(side=tk.LEFT)
        button2 = tk.Button(self,
                            text="REDO",
                            command=lambda: self.start(True),
                            font=tkFont.Font(family="Calibri", size=12))
        button2.pack(side=tk.RIGHT)

    def filter_condition(self, cell):
        if self.o4neg.get() and "neg" in cell:
            return True
        if self.o4pos.get() and "pos" in cell:
            return True
        if self.unclassified.get() and "unknown" in cell:
            return True
        return False

    def runAnnotation(self, name, redo):
        """ Run main annotation algorithm in selected folder."""

        folder_root = settings.experiments[name]['root']
        o4_gamma = settings.experiments[name]['o4_gamma']

        keras_folder = os.path.join(folder_root, 'keras')

        # load annotations
        filename = fullPath(folder_root, 'annotations.json')
        with open(filename, 'r') as f:
            annotations = json.load(f)

        def badAnnotation(cell, newAnnotation):
            """Deal with file with bad annotation."""
            # print(cell['cell'])
            original = cell['cell']
            replacement = 'unknown'
            if newAnnotation == 0:
                replacement = 'o4neg'
            elif newAnnotation == 1:
                replacement = 'o4pos'

            import re
            new = re.sub(r'unknown|o4pos|o4neg', replacement, original)

            if not os.path.isfile(fullPath(keras_folder, original)):
                # file not found
                print(f"File {original} not found!")
                return

            os.rename(fullPath(keras_folder, original), fullPath(keras_folder, new))
            print(f"File {original} renamed to {new}.")

            return new

        def checkKey():
            """ Wait for user response."""
            flag = True
            while flag:
                k = cv.waitKey()
                if k == 113:
                    # if 'q' then break for loop
                    return True
                elif k == 121:
                    # if 'y' (O4 pos cell) then
                    if cell['classification'] == 1:
                        cell['annotation'] = 1
                    elif cell['classification'] != 1:
                        cell['annotation'] = 0
                        cell['cell'] = badAnnotation(cell, 1)
                    flag = False
                elif k == 110:
                    # if 'n' (O4 neg cell) then
                    if cell['classification'] == 0:
                        cell['annotation'] = 0
                    elif cell['classification'] != 1:
                        cell['annotation'] = 1
                        cell['cell'] = badAnnotation(cell, 0)
                    flag = False
            return False

        for cell in annotations:
            print(cell['cell'])

            # filter based on type (using file name)
            if not self.filter_condition(cell['cell']):
                continue

            # skip if annotation has been set and redo was not checked
            if 'annotation' in cell and not redo:
                continue

            # read image
            img = cv.imread(fullPath(keras_folder, cell['cell']))
            if img is None:
                print("Could not open:", cell['cell'])
                cell['annotation'] = -1
                continue

            # images are saved with DAPI in red
            b, g, r = cv.split(img)
            g = singleCompositeImage.gammaCorrect(g, o4_gamma)
            g = cv.normalize(src=g, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            b = cv.normalize(src=b, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            img = cv.merge((r, g, b))

            # setup window
            cv.namedWindow(cell['cell'], cv.WINDOW_AUTOSIZE)
            cv.moveWindow(cell['cell'], 100, 100)

            # increase scale to improve visibility
            scale_percent = 500  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

            # place dot on cell-of-interest
            color = (255, 0, 0)
            img = cv.circle(img, (int(width / 2), int(height / 2)), 10, color, -1)

            # label image
            img = cv.putText(img,
                             cell['cell'],
                             (10, 40),
                             fontFace=cv.FONT_HERSHEY_SIMPLEX,
                             fontScale=1,
                             color=(255, 255, 255))

            # show image
            cv.imshow(cell['cell'], img)

            # check for key press
            if checkKey():
                break

            # clean-up
            cv.destroyAllWindows()

        # save annotations
        filename = os.path.join(folder_root, 'annotations.json')
        with open(filename, 'w') as f:
            json.dump(annotations, f)

        print("Finished updating annotations.json")


# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)
app = Application(master=root)
app.mainloop()
