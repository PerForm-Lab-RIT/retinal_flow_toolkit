@echo off
call conda activate py38
jupyter notebook --notebook-dir="%cd%"