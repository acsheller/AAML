import streamlit as st
from kubernetes import client, config 

def set_config():
    '''
    Return it from one spot so only 
    need to change it in one spot.
    '''
    config.load_incluster_config()
    #config.load_kube_config()




# Sidebar
st.sidebar.title("Configuration")

# Add widgets to the sidebar
user_input = st.sidebar.text_input("Enter something:")
selected_option = st.sidebar.selectbox("Select an option:", ["Option 1", "Option 2", "Option 3"])
button_clicked = st.sidebar.button("Click Me")

# Main Content
st.title("DRL Agent as a Kubernetes Scheduler")
st.subheader('EN.705.742 Advanced Applied Machine Learning')

st.markdown(
"""
        **Overview**
    
        Information about GPU. this uses the Python Moduel 
        nvsmi 
     
        """)



# Display user input and selected option in the main content area
st.write(f"You entered: {user_input}")
st.write(f"Selected option: {selected_option}")

# Display a message when the button is clicked
if button_clicked:
    st.success("Button clicked!")

# You can add more content and visualizations in the main content area.
# For example:
# st.plotly_chart(your_plotly_figure)
# st.pyplot(your_matplotlib_plot)
# st.dataframe(your_dataframe)
