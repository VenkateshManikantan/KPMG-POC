import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from tools.fileingestor import FileIngestor

# Set the title for the Streamlit app


# Create a file uploader in the sidebar


with st.sidebar:
    st.image(r'images/kpmg-logo.svg')
    st.title('KPMG - POC')
    st.markdown('''
    ## Functionality:
    Testing multiple POC with respect to interacting 
    with contract document.
                    
    Tech Stack:
    - streamlit
    - Langchain
    - OpenAI
    - Llamma

    ## Developer:

    - Venkatesh Manikantan -> Assistant Manager KPMG India 
    - Digital LightHouse
     
    ''')

    add_vertical_space(4)
    st.write('© 2023 KPMG Assurance and Consulting Services LLP, an Indian Limited Liability Partnership and a member firm of the KPMG global organization of independent member firms')
    st.write('affiliated with KPMG International Limited, a private English company limited by guarantee. All rights reserved.For more detail about the structure of the KPMG global organization please visit https://kpmg.com/governance.')



def main():
    st.header("📄 Interact with your contract")
    uploaded_file = st.file_uploader("Upload File", type="pdf")

    if uploaded_file:
        file_ingestor = FileIngestor(uploaded_file)
        file_ingestor.handlefileandingest()



if __name__=="__main__":
    main()