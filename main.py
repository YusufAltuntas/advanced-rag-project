from dotenv import load_dotenv
from graph.graph import app
load_dotenv()

# if __name__ == "__main__":
#     print(app.invoke(input={"question": "What is prompt engineering?"}))


#def invoke_question(question):
#    return app.invoke(input={"question": question})
#
#if __name__ == "__main__":
#    result = invoke_question("What is prompt engineering?")
#    print(result)
#
import gradio as gr

# Define your function
def invoke_question(question):
    result = app.invoke(input={"question": question})
    return result['generation']
    
# Create the Gradio interface
iface = gr.Interface(
    fn=invoke_question,           # Function to be executed
    inputs=gr.Textbox(),           # Input component for the question
    outputs=gr.Textbox(),          # Output component for the result
    title="Invoke Question",       # Title of the interface
    description="Ask a question"   # Description
)

# Launch the interface
iface.launch()