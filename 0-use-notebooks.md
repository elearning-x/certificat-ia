---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: Jupyter Notebook
  version: '3.0'
---

<div class="licence">
<span>Licence CC BY-NC-ND</span>
<span>Thierry Parmentelat &amp; Arnaud Legout</span>
<span><img src="media/both-logos-small-alpha.png" /></span>
</div>

+++

# Jupyter notebooks as course material

+++

To illustrate the MOOC videos, we have chosen to use Jupyter to write "mixed" documents containing text and Python code, called "notebooks", of which the present document is an example.

In the following, we will use Python code, but we have not yet learned about the language. Don't worry, this code is only intended to validate the operation of notebooks, and we only use very simple things.

+++

### Warning: browser settings

+++

First of all, for a good behavior of the notebooks, we remind you that it is necessary to have **authorized** in your browser the **cookies** coming from the website **`nbhosting.inria.fr`**, which hosts the infrastructure that hosts all the notebooks.

+++

### Advantages of notebooks

+++

As you can see, this support allows a more readable format than comments in a code file.

+++

Please note that **the code fragments can be evaluated and modified**. So you can easily try out variants based on the original notebook.

Also note that the Python code is interpreted **on a remote machine**, which allows you to make your first attempts before you have even installed Python on your own computer.

+++

### How to use notebooks

+++

At the top of the notebook, you have a menu bar (on a light blue background), containing:

* a title for the notebook, with a version number ;
* a menu bar with the entries `File`, `Insert`, `Cell`, `Kernel`;
* and a button bar which are shortcuts to some frequently used menus. If you leave your mouse over a button, a small text appears, indicating which function the button corresponds to.

We saw in the video that a notebook is made up of a series of cells, either textual or containing code. The code cells are easily recognizable, they are preceded by `In [ ]:`. The cell after the one you are reading is a code cell.

To begin, select that code cell with your mouse, and press the one in the menu bar - at the top of the notebook, that is - the one shaped like a triangular arrow to the right (Play):
<img src="media/notebook-eval-button.png">

```{code-cell} ipython3
---
vscode:
  languageId: python
---
20 * 30
```

As you can see, the cell is "executed" (we would rather say evaluated), and we move to the next cell.

Alternatively, you can simply type ***Shift+Enter***, to get the same effect. Generally speaking, it is important to learn and use keyboard shortcuts, as this will save you a lot of time later on.

+++

La façon habituelle d'*exécuter* l'ensemble du notebook consiste :

* à sélectionner la première cellule,
* et à taper ***Shift+Enter*** jusqu'à atteindre la fin du notebook.

+++

When a code cell has been evaluated, Jupyter adds an `Out` cell below the `In` cell that gives the result of the Python fragment, i.e. above 600.

Jupyter also adds a number between the square brackets to show, for example above, `In [1]:`. This number allows you to find the order in which the cells were evaluated.

+++

Of course, you can modify these code cells for testing purposes; for example, you can use the model below to calculate the square root of 3, or try the function on a negative number and see how the error is reported.

```{code-cell} ipython3
---
vscode:
  languageId: python
---
# math.sqrt (pour square root) calcule la racine carrée
import math
math.sqrt(2)
```

You can also evaluate the whole notebook at once by using the *Cell -> Run All* menu.

+++

### Be careful to evaluate the cells in the right order

+++

It is important that the code cells are evaluated in the correct order. If you don't respect the order in which the code cells are presented, the result may be unexpected.

In fact, evaluating a program in notebook form is like cutting it into small fragments, and if you run these fragments out of order, you will naturally get a different program.

+++

We can see it on this example:

```{code-cell} ipython3
---
vscode:
  languageId: python
---
message = "Pay attention to the order in which you evaluate the notebooks"
```

```{code-cell} ipython3
---
vscode:
  languageId: python
---
print(message)
```

If a little further in the notebook we do for example :

```{code-cell} ipython3
---
vscode:
  languageId: python
---
# this will clear the 'message' variable
del message
```

which makes the `message` symbol undefined, then of course we can no longer evaluate the cell that does `print` since the `message` variable is no longer known to the interpreter.

+++

### Reset the interpreter

+++

If you make too many changes, or lose track of what you've been evaluating, it can be useful to restart your interpreter. The *Kernel → Restart* menu allows you to do this, much in the way that IDLE restarts from a blank interpreter when you use the F5 function.

+++

The *Kernel → Interrupt* menu, on the other hand, can be used if your fragment takes too long to execute (for example you have written a loop whose logic is broken and does not finish).

+++

### You are working on a copy

+++

One of the main advantages of notebooks is that they allow you to modify the code we have written, and see for yourself how the modified code performs.

For this reason, each student has their **own copy** of each notebook, so of course you can make any changes you want to your notebooks without affecting the other students.

+++

### Go back to the course version

+++

You can always go back to the "course" version with the menu
*File → Reset to original*.

+++

Be careful, with this function you restore **all the notebook** and therefore **you lose your modifications on this notebook**.

+++

### Download in Python format

+++

You can download a notebook in Python format to your computer using the menu
*File → Download as → Python*

+++

Text cells are preserved in the result as Python comments.

+++

### Share a read-only notebook

+++

Finally, with the *File → Share static version* menu, you can publish a read-only version of your notebook; you get a URL that you can post, for example to ask for help on the forum. This way, other students can have read-only access to your code.

+++

Note that when you use this feature multiple times, your classmates will always see the latest published version, the URL used is always the same for a given student and notebook.

+++

### Add cells

+++

You can add a cell anywhere in the document with the **+** button on the button bar.

Also, when you get to the end of the document, a new cell is created each time you evaluate the last cell, so you have a draft for your own testing.

Now it is your turn.
