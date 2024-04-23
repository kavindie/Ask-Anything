import gradio as gr


def sentence_builder(expertise, option, backpack, confidence):
    #return f"""{"I give consent" if yes else "I do not give consent"}. I am {expertise}. I could not think of {option}s. The sttetemnt there is a red backpack is {backpack}. I am {confidence} confident"""
    return f"""I am {expertise}. I could not think of {option}s. The sttetemnt there is a red backpack is {backpack}. I am {confidence} confident"""

def video_identity(video):
    return video

def prepare_next_tab():
    # Here you simulate preparing or activating the next tab content
    # Since direct tab manipulation isn't supported, consider this as a placeholder
    # for any preparation or activation logic
    print("Preparing the next tab...")
    with gr.Tab("Test"):
        with gr.Row():
            vid1 = gr.PlayableVideo(interactive=True)


with gr.Blocks() as app:
    with gr.Tab("Description") as tab_des:
        gr.Markdown("Here goes the description of the user study and links to ethical clearance forms. Once you click yes, you cannot go back.")
        yes = gr.Checkbox(label="Yes", info="I have read the privacy notes")
        # with gr.Row(visible=False) as expert_row:
        #     expertise = gr.Dropdown(
        #         value=["Technical-expert", "Domain-expert", "Neither technical nor domain expert", "Technical and domain expert"], 
        #             label="Expertise", 
        #             info="How much are you aware of robotics operator tasks and/or video summarization",
        #         )
        error_box = gr.Textbox(value="You must agree to continue", label="Error", visible=False)

    with gr.Tab("Participant Information", visible=False) as tab_participant:
        gr.Markdown("Here goes the details of the user")
        name = gr.Textbox(label="name", info="Give an annonymous name")
        expertise = gr.Dropdown(
                ["Technical-expert", "Domain-expert", "Neither technical nor domain expert", "Technical and domain expert"], label="Expertise?", info="How much are you aware of robotics operator tasks and/or video summarization"
            )
        gr.Markdown("Once you click Next, you cannot go back.")
        next_button = gr.Button("Next")


    with gr.Tab("Raw Videos", visible=False) as tab_raw_video:
        with gr.Row():
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/yoga.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/jesse_dance.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/0tmA_C6XwfM.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/yoga.mp4", show_label=False)
            # vid1 = gr.PlayableVideo(interactive=True)
            # vid2 = gr.PlayableVideo(interactive=True)
            # vid3 = gr.PlayableVideo(interactive=True)
            # vid4 = gr.PlayableVideo(interactive=True)
            # vid1.change(lambda: print("change"))
            # vid1.clear(lambda: print("clear"))
            # vid1.start_recording(lambda: print("start_recording"))
            # vid1.stop_recording(lambda: print("stop_recording"))
            # vid1.stop(lambda: print("stop"))
            # vid1.play(lambda: print("play"))
            # vid1.pause(lambda: print("pause"))
            # vid1.end(lambda: print("end"))
            # vid1.upload(lambda: print("upload"))

        with gr.Column():
            gr.Markdown("Is there a backpack in the given videos?")
            backpack = gr.Radio(["True", "False"], label="Choose one")
            backpack_colour = gr.Textbox(label="Colour", info="What is the colour of the backpack?", visible=False)
            gr.Markdown("Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each?")
            gr.Radio(
                label="Drill",
                choices=["Drill: 0", "Drill: 1-3", "Drill: >3"],
                interactive=True
            )
            gr.Radio(
                label="Blue Rope",
                choices=["Blue Rope: 0", "Blue Rope: 1-3", "Blue Rope: >3"],
                interactive=True
            )
            gr.Radio(
                label="Fire Extinguisher",
                choices=["Fire Extinguisher: 0", "Fire Extinguisher: 1-3", "Fire Extinguisher: >3"],
                interactive=True
            )
            gr.Radio(
                label="Cell Phone",
                choices=["Cell Phone: 0", "Cell Phone: 1-3", "Cell Phone: >3"],
                interactive=True
            )
            
            confidence = gr.Slider(1, 10, interactive=True, value=4, label="Confidence", info="How confident are you about the above answerss? 1- Not confident at all, 10- Very confident")

            gr.Markdown("Once you click Submit, you cannot go back.")
            button_1 = gr.Button("Submit")


    with gr.Tab("Generic Summary Videos", visible=False) as tab_gen_sum:
        with gr.Row():
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/0tmA_C6XwfM.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/jesse_dance.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/yoga.mp4", show_label=False)
            gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/yoga.mp4", show_label=False)

        with gr.Column():
            gr.Markdown("Is there a backpack in the given videos?")
            backpack = gr.Radio(["True", "False"], label="Choose one")
            backpack_colour = gr.Textbox(label="Colour", info="What is the colour of the backpack?", visible=False)
            gr.Markdown("Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each?")
            gr.Radio(
                label="Drill",
                choices=["Drill: 0", "Drill: 1-3", "Drill: >3"],
                interactive=True
            )
            gr.Radio(
                label="Blue Rope",
                choices=["Blue Rope: 0", "Blue Rope: 1-3", "Blue Rope: >3"],
                interactive=True
            )
            gr.Radio(
                label="Fire Extinguisher",
                choices=["Fire Extinguisher: 0", "Fire Extinguisher: 1-3", "Fire Extinguisher: >3"],
                interactive=True
            )
            gr.Radio(
                label="Cell Phone",
                choices=["Cell Phone: 0", "Cell Phone: 1-3", "Cell Phone: >3"],
                interactive=True
            )
            
            confidence = gr.Slider(1, 10, interactive=True, value=4, label="Confidence", info="How confident are you about the above answerss? 1- Not confident at all, 10- Very confident")

            gr.Markdown("Once you click Submit, you cannot go back.")
            button_2 = gr.Button("Submit")

    with gr.Tab("Query Driven Summary Videos", visible=False) as tab_que_sum:
        #Todo: need to combine the 4 videos and insert
        gr.Markdown("I am here")
        with gr.Row():
            with gr.Column(visible=True):
                gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/yoga.mp4", show_label=False)
            with gr.Column(visible=True):
                chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')


    def consent_yes(yes):
        if not yes:
            return {
                tab_participant: gr.Tab(visible=False),
                error_box: gr.Textbox(visible=True),
                tab_des: gr.Tab(visible=True)
                }
        else:
            return {
                error_box: gr.Textbox(visible=False),
                tab_participant: gr.Tab(visible=True),
                tab_des: gr.Tab(visible=False),
            }  
    
    yes.select(
        consent_yes,
        [yes],
        [error_box, tab_participant, tab_des]   
    )

    def submit_tab_participant():
        return {
            tab_raw_video: gr.Tab(visible=True),
            tab_participant: gr.Tab(visible=False)
        } 
    
    next_button.click(
        submit_tab_participant, 
        inputs=[], 
        outputs=[tab_raw_video, tab_participant]
    )

    def submit_tab_raw_video():
        return {
            tab_gen_sum: gr.Tab(visible=True),
            tab_raw_video:gr.Tab(visible=False)
        } 
    
    button_1.click(
        submit_tab_raw_video, 
        inputs=[], 
        outputs=[tab_gen_sum,tab_raw_video]
    )

    def submit_tab_gen_video():
        return {
            tab_que_sum: gr.Tab(visible=True),
            tab_gen_sum:gr.Tab(visible=False)
        } 
    
    button_2.click(
        submit_tab_gen_video, 
        inputs=[], 
        outputs=[tab_gen_sum,tab_que_sum]
    )

app.launch()




    






# demo = gr.Interface(
#     sentence_builder,
#     [
#         gr.Checkbox(label="Yes", info="I have read the privacy notes"),
#         gr.Dropdown(
#             ["Techincal-expert", "Domain-expert", "Neither technical nor domain expert", "Technical and domain expert"], label="Expertise?", info="How much are you aware of robotics operator tasks and/or video summarization"
#         ),
#         gr.CheckboxGroup(["Option 1", "Option 2", "Option 3"], label="Option?", info="To enable selecting multiple"),
#         gr.Radio(["True", "False"], label="Backpack", info="Is there a red backpack in the video?"),
#         gr.Slider(1, 10, value=4, label="Confidence", info="How confident are you about the answer? 1- Not confident at all, 10- Very confident"),
#         # gr.Dropdown(
#         #     ["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity", info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
#         # ),
#     ],
#     "text",
#     examples=[
#         [True, "Techincal-expert", ["Option 1", "Option 2"], "True", 2],
#         [False, "Neither technical nor domain expert", ["Option 1"], "Fasle", 9],
#     ]
# )

# if __name__ == "__main__":
#     demo.launch()
