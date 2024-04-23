import gradio as gr

# Define the function to show upon submission
# Note: The example function here just echoes the inputs for demonstration purposes.
# You might replace it with your actual logic.
def process_responses():
    return None

# Define your inputs
backpack = gr.Radio(["True", "False"], label="Backpack", info="Is there a backpack in the given videos?")
backpack_colour = gr.Textbox(label="Colour", info="What is the colour of the backpack?", visible=False)
other_objs = gr.CheckboxGroup(
    [
        "Drill: 0", "Drill: 1-3", "Drill: >3",
        "Blue Rope: 0", "Blue Rope: 1-3", "Blue Rope: >3",
        "Fire Extinguisher: 0", "Fire Extinguisher: 1-3", "Fire Extinguisher: >3",
        "Cell Phone: 0", "Cell Phone: 1-3", "Cell Phone: >3",
    ],
    label="Objects",
    info="Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each?")
confidence = gr.Slider(1, 10, value=4, label="Confidence", info="How confident are you about the answer? 1- Not confident at all, 10- Very confident")

# Define function and inputs for each tab
tab_inputs = [backpack, backpack_colour, other_objs, confidence]
tab_outputs = gr.JSON()

# Create a Gradio Interface
with gr.Blocks() as tabs_app:
    gr.Markdown("## Video Analysis Tabs")

    with gr.Tab("Video 1"):
        gr.Interface(fn=process_responses, inputs=tab_inputs, outputs=[]).render()

    with gr.Tab("Video 2"):
        gr.Interface(fn=process_responses, inputs=tab_inputs, outputs=tab_outputs).render()

    with gr.Tab("Video 3"):
        gr.Interface(fn=process_responses, inputs=tab_inputs, outputs=tab_outputs).render()

# Launch the Gradio app
tabs_app.launch()