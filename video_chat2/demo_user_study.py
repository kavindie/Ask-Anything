import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from conversation import Chat

# videochat
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat2_it import VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType


# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)
    cfg.model.vision_encoder.num_frames = 4
    # cfg.model.videochat2_model_path = ""
    # cfg.model.debug = True
    model = VideoChat2_it(config=cfg.model)
    model = model.to(torch.device(cfg.device))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )
    model.llama_model = get_peft_model(model.llama_model, peft_config)
    state_dict = torch.load("./videochat2_7b_stage3.pth", "cpu")
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.eval()

    chat = Chat(model)
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img, gr_video, chat_state, num_segments):
    print(gr_img, gr_video)
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return llm_message, img_list, chat_state
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask_answer(user_message, history):
    global chat_state, img_list, llm_message
    history = chat_state
    chat_state =  chat.ask(user_message, chat_state)
    llm_message, llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=1, temperature=1)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    print(chat_state)
    print(f"Answer: {llm_message}")
    yield llm_message

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state =  chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )


title = "User Study - Need CSIRO logo or something"
description = "This is a user study - a description of the study maybe"

def sentence_builder(yes, expertise, option, backpack, confidence):
    return f"""{"I give consent" if yes else "I do not give consent"}. I am {expertise}. I could not think of {option}s. The sttetemnt there is a red backpack is {backpack}. I am {confidence} confident"""


with gr.Blocks(title="Video Summarization!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

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
            with gr.Column():
                gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/camera_long_6.mp4", show_label=False)

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
            with gr.Column():
                gr.Video(value="/scratch3/kat049/Ask-Anything/video_chat2/example/output_6.mp4", show_label=False)

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
        num_segments = 80
        video_path = '/scratch3/kat049/Ask-Anything/video_chat2/example/output_6/final_video_barrel.mp4'
        
        global chat_state, img_list, llm_message
        chat_state = gr.State()
        img_list = gr.State()
        chat = init_model()
        llm_message, img_list, chat_state = upload_img(None, video_path, chat_state, num_segments)

        with gr.Row():
            with gr.Column():
                gr.Interface(
                    fn=gradio_ask_answer,
                    inputs='text',
                    outputs='text',
                )
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
                button_3 = gr.Button("Finish")
        

    # with gr.Tab("Query Driven Summary Videos", visible=False) as tab_que_sum_old:
    #     with gr.Row():
    #         with gr.Column(scale=0.5, visible=True) as video_upload:
    #             with gr.Column(elem_id="image", scale=0.5) as img_part:
    #                 with gr.Tab("Video", elem_id='video_tab'):
    #                     up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload", height=360)
    #                 with gr.Tab("Image", elem_id='image_tab'):
    #                     up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload", height=360)
    #             upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
    #             clear = gr.Button("Restart")
                
    #             num_beams = gr.Slider(
    #                 minimum=1,
    #                 maximum=10,
    #                 value=1,
    #                 step=1,
    #                 interactive=True,
    #                 label="beam search numbers)",
    #             )
                
    #             temperature = gr.Slider(
    #                 minimum=0.1,
    #                 maximum=2.0,
    #                 value=1.0,
    #                 step=0.1,
    #                 interactive=True,
    #                 label="Temperature",
    #             )
                
    #             num_segments = gr.Slider(
    #                 minimum=8,
    #                 maximum=80,
    #                 value=8,
    #                 step=1,
    #                 interactive=True,
    #                 label="Video Segments",
    #             )
            
    #         with gr.Column(visible=True)  as input_raws:
    #             chat_state = gr.State()
    #             img_list = gr.State()
    #             chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
    #             with gr.Row():
    #                 with gr.Column(scale=0.7):
    #                     text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False, container=False)
    #                 with gr.Column(scale=0.15, min_width=0):
    #                     run = gr.Button("💭Send")
    #                 with gr.Column(scale=0.15, min_width=0):
    #                     clear = gr.Button("🔄Clear️")     
        
    #     chat = init_model()
    #     upload_button.click(upload_img, [up_image, up_video, chat_state, num_segments], [up_image, up_video, text_input, upload_button, chat_state, img_list])
        
    #     text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
    #         gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    #     )
    #     run.click(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
    #         gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    #     )
    #     run.click(lambda: "", None, text_input)  
    #     clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_image, up_video, text_input, upload_button, chat_state, img_list], queue=False)
        
        # with gr.Column():
        #     gr.Markdown("Is there a backpack in the given videos?")
        #     backpack = gr.Radio(["True", "False"], label="Choose one")
        #     backpack_colour = gr.Textbox(label="Colour", info="What is the colour of the backpack?", visible=False)
        #     gr.Markdown("Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each?")
        #     gr.Radio(
        #         label="Drill",
        #         choices=["Drill: 0", "Drill: 1-3", "Drill: >3"],
        #         interactive=True
        #     )
        #     gr.Radio(
        #         label="Blue Rope",
        #         choices=["Blue Rope: 0", "Blue Rope: 1-3", "Blue Rope: >3"],
        #         interactive=True
        #     )
        #     gr.Radio(
        #         label="Fire Extinguisher",
        #         choices=["Fire Extinguisher: 0", "Fire Extinguisher: 1-3", "Fire Extinguisher: >3"],
        #         interactive=True
        #     )
        #     gr.Radio(
        #         label="Cell Phone",
        #         choices=["Cell Phone: 0", "Cell Phone: 1-3", "Cell Phone: >3"],
        #         interactive=True
        #     )
            
        #     confidence = gr.Slider(1, 10, interactive=True, value=4, label="Confidence", info="How confident are you about the above answerss? 1- Not confident at all, 10- Very confident")

        #     gr.Markdown("Once you click Submit, you cannot go back.")
        #     button_3 = gr.Button("Finish")
        
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



demo.queue()
demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=10034)
