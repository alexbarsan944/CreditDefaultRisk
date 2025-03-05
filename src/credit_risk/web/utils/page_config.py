import streamlit as st

def set_page_config(page_title: str = "Credit Risk Prediction"):
    """
    Configure the Streamlit page.
    
    Parameters
    ----------
    page_title : str
        Title of the page
    """
    st.set_page_config(
        page_title=f"Credit Risk - {page_title}",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    ) 