# Shell

## Create .gitignore file from shell/git bash
`touch .gitignore`

## Quickly add line to existing .gitignore
`echo '.DS_Store' >> .gitignore`

### Add .ipnyb_checkpoints folder (Jupyter Notebook) to .gitingore
`echo '.ipynb_checkpoints/' >> .gitignore`

# Anaconda/Conda

## Export active conda environment to new file
`conda env export > environment.yml`

## Create conda environment from existing environment.yml file
`conda env create -f environment.yml`

## Create an environment with a specific package
`conda create -n myenv scipy`

## List all installed/local conda environments
`conda env list`

## Create exact copy of an existing local environment
`conda create --name myclone --clone myenv`

## Generate a requirements file and then install from it in another environment
`pip freeze > requirements.txt`
`pip install -r requirements.txt`

## Update Conda and Anaconda from older versions

Note: Make sure terminal has administrative rights on windows/sudo on linux before running the following commands:

`conda update conda`
`conda update anaconda`

# Jupyter Notebook

`ESC` -> `R` -> `Y` - Make cell Raw, Then make it a code cell again (for quick output cell clear)

## Command Mode (Blue)

`H`- Show all Keyboard shortcuts

`L` - Toggle line numbers

`Shift + Enter` run current cell + insert new cell below, stay in command mode

`CTRL + Enter` run current cell 

`Alt + Enter` run current cell + insert new cell bellow, enter edit mode

`Shift + Tab` - Docstring for current cell/command

`ESC + O` - Toggle cell output

`ESC + F` - Find/search on cells/code but not output cells

`Shift + M` with multiple cells selected - Merge cells

`D +D ` - Press D twice to delete current cell

`A` - insert new cell above current cell

`B` - insert new cell below current cell

## Edit Mode (Green)

`Ctrl + /` while selecting lines/cells - Block comment/uncomment

`Click + Drag while holding down Alt` - Multicursor 
