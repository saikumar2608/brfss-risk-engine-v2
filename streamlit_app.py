# streamlit_app.py  
import sys, os
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.append(root)

#  real app 
import app.app  
