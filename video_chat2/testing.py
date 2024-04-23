from utils.config import Config
from models.videochat2_it import VideoChat2_it
import torch
from peft import get_peft_model, LoraConfig, TaskType
from conversation import Chat
from utils.easydict import EasyDict
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import PILToTensor


def load_models():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)

    # load stage2 model
    cfg.model.vision_encoder.num_frames = 4
    # cfg.model.videochat2_model_path = ""
    # cfg.model.debug = True
    model = VideoChat2_it(config=cfg.model)
    model = model.to(torch.device(cfg.device))

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )
    model.llama_model = get_peft_model(model.llama_model, peft_config)
    
    state_dict = torch.load("./videochat2_7b_stage3.pth", cfg.device)
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.eval()

    chat = Chat(model, device=cfg.device)
    print('Initialization Finished')
    return chat


def process_image(image_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    resolution = 224
    
    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    # No need of random transform if evaluating
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    image = test_transform(image)
    return image

    

def process_video_copied():
    num_frames = 8
    num_frames_test = 8
    batch_size = 4
    max_txt_l = 512

    pre_text = False

    inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
    )


    video_reader_type = inputs.video_input.get("video_reader_type", "decord")
    video_only_dataset_kwargs_train = dict(
        video_reader_type=video_reader_type,
        sample_type=inputs.video_input.sample_type,
        num_frames=inputs.video_input.num_frames,
        num_tries=3,  # false tolerance
    )
    video_only_dataset_kwargs_eval = dict(
        video_reader_type=video_reader_type,
        sample_type=inputs.video_input.sample_type_test,
        num_frames=inputs.video_input.num_frames_test,
        num_tries=1,  # we want to have predictions for all videos
    )
    for _ in range(self.num_tries):
        try:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            frames, frame_indices, fps = self.video_reader(
                data_path, self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client, clip=clip
            )
        except Exception as e:
            logger.warning(
                f"Caught exception {e} when loading video {data_path}, "
                f"randomly sample a new video as replacement"
            )
            index = random.randint(0, len(self) - 1)
            ann = self.get_anno(index)
            data_path = ann["image"]
            continue
        # shared aug for video frames
        frames = self.transform(frames)
        if return_fps:
            sec = [str(round(f / fps, 1)) for f in frame_indices]
            return frames, index, sec
        else:
            return frames, index
    else:
        raise RuntimeError(
            f"Failed to fetch video after {self.num_tries} tries. "
            f"This might indicate that you have many corrupted videos."
        )


def gradio_stuff(chat):
    # # Let's first focus on the image
    # image_path = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/TestDataset_11/Images/frame_5.jpg'
    # image_path = '/scratch2/kat049/Git/Ask-Anything/video_chat2/example/frame_2797.jpg'
    # image = process_image(image_path)
    # #image = Image.open(image_path)
    # img_list = []
    # llm_message, img_list,chat_state = chat.upload_img(image_path, chat_state, img_list)
    # user_message = "Provide a brief description of the given image."

    # Video now
    #video_path = '/scratch2/kat049/Git/CGDETR/run_on_video/example/output.mp4'
    user_messages = [
        "Is there a backpack in the given videos?",
        "If there is a backpack in the video, what is the color of the backpack?",
        "Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each item can you observe?",
        "Based on the video, where do you think the camera mounted on the robot: front/right/left/back?",
        "Was the robot stuck at any point during the run?",
        "Was the robot idling during the run?",
    ]
    for vid in range(2, 8):
        chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        video_path = f'/scratch3/kat049/Ask-Anything/video_chat2/example/output_{vid}.avi'
        #video_path = '/scratch2/kat049/Git/Ask-Anything/video_chat2/example/yoga.mp4'
        num_segments = 100
        img_list = []
        llm_message, img_list, chat_state = chat.upload_video(video_path, chat_state, img_list, num_segments)
        
        for user_message in user_messages:

            chat_state =  chat.ask(user_message, chat_state)
            
            llm_message,llm_message_token, chat_state = chat.answer(
                conv=chat_state, 
                img_list=img_list, 
                do_sample=False, 
                max_new_tokens=1000, 
                num_beams=5,
                #min_length=1,
                #top_p=0.9,
                #repetition_penalty=1.0, 
                #length_penalty=1,
                #temperature=0.5
            )
            llm_message = llm_message.replace("<s>", "") # handle <s>

            with open('/scratch3/kat049/Ask-Anything/video_chat2/example/llm_messages.txt', 'a') as file:
                file.write(llm_message + '\n')

    print(f"Answer: {llm_message}")



if __name__== '__main__':
    chat = load_models()
    gradio_stuff(chat)
    
from utils.config import Config
from models.videochat2_it import VideoChat2_it
import torch
from peft import get_peft_model, LoraConfig, TaskType
from conversation import Chat
from utils.easydict import EasyDict
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import PILToTensor


def load_models():
    print('Initializing VideoChat')
    config_file = "configs/config.json"
    cfg = Config.from_file(config_file)

    # load stage2 model
    cfg.model.vision_encoder.num_frames = 4
    # cfg.model.videochat2_model_path = ""
    # cfg.model.debug = True
    model = VideoChat2_it(config=cfg.model)
    model = model.to(torch.device(cfg.device))

    # add lora to run stage3 model
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, 
        r=16, lora_alpha=32, lora_dropout=0.
    )
    model.llama_model = get_peft_model(model.llama_model, peft_config)
    
    state_dict = torch.load("./videochat2_7b_stage3.pth", cfg.device)
    if 'model' in state_dict.keys():
        msg = model.load_state_dict(state_dict['model'], strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.eval()

    chat = Chat(model, device=cfg.device)
    print('Initialization Finished')
    return chat


def process_image(image_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    resolution = 224
    
    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    # No need of random transform if evaluating
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image).unsqueeze(0)  # (1, C, H, W), torch.uint8
    image = test_transform(image)
    return image

    

def process_video_copied():
    num_frames = 8
    num_frames_test = 8
    batch_size = 4
    max_txt_l = 512

    pre_text = False

    inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
    )


    video_reader_type = inputs.video_input.get("video_reader_type", "decord")
    video_only_dataset_kwargs_train = dict(
        video_reader_type=video_reader_type,
        sample_type=inputs.video_input.sample_type,
        num_frames=inputs.video_input.num_frames,
        num_tries=3,  # false tolerance
    )
    video_only_dataset_kwargs_eval = dict(
        video_reader_type=video_reader_type,
        sample_type=inputs.video_input.sample_type_test,
        num_frames=inputs.video_input.num_frames_test,
        num_tries=1,  # we want to have predictions for all videos
    )
    for _ in range(self.num_tries):
        try:
            max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
            frames, frame_indices, fps = self.video_reader(
                data_path, self.num_frames, self.sample_type, 
                max_num_frames=max_num_frames, client=self.client, clip=clip
            )
        except Exception as e:
            logger.warning(
                f"Caught exception {e} when loading video {data_path}, "
                f"randomly sample a new video as replacement"
            )
            index = random.randint(0, len(self) - 1)
            ann = self.get_anno(index)
            data_path = ann["image"]
            continue
        # shared aug for video frames
        frames = self.transform(frames)
        if return_fps:
            sec = [str(round(f / fps, 1)) for f in frame_indices]
            return frames, index, sec
        else:
            return frames, index
    else:
        raise RuntimeError(
            f"Failed to fetch video after {self.num_tries} tries. "
            f"This might indicate that you have many corrupted videos."
        )


def gradio_stuff(chat):
    # # Let's first focus on the image
    # image_path = '/scratch2/kat049/Git/STVT/STVT/STVT/datasets/TestDataset_11/Images/frame_5.jpg'
    # image_path = '/scratch2/kat049/Git/Ask-Anything/video_chat2/example/frame_2797.jpg'
    # image = process_image(image_path)
    # #image = Image.open(image_path)
    # img_list = []
    # llm_message, img_list,chat_state = chat.upload_img(image_path, chat_state, img_list)
    # user_message = "Provide a brief description of the given image."

    # Video now
    #video_path = '/scratch2/kat049/Git/CGDETR/run_on_video/example/output.mp4'
    user_messages = [
        "Is there a backpack in the given videos?",
        "If there is a backpack in the video, what is the color of the backpack?",
        "Out of the following list, what can you observe {drill, blue rope, fire extinguisher, cell phone} and how many of each item can you observe?",
        "Based on the video, where do you think the camera mounted on the robot: front/right/left/back?",
        "Was the robot stuck at any point during the run?",
        "Was the robot idling during the run?",
    ]
    for vid in range(2, 8):
        chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })

        video_path = f'/scratch2/kat049/Git/Ask-Anything/video_chat2/example/output_{vid}.avi'
        #video_path = '/scratch2/kat049/Git/Ask-Anything/video_chat2/example/yoga.mp4'
        num_segments = 50
        img_list = []
        llm_message, img_list, chat_state = chat.upload_video(video_path, chat_state, img_list, num_segments)
        
        for user_message in user_messages:

            chat_state =  chat.ask(user_message, chat_state)
            
            llm_message,llm_message_token, chat_state = chat.answer(
                conv=chat_state, 
                img_list=img_list, 
                do_sample=False, 
                max_new_tokens=1000, 
                num_beams=1,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0, 
                length_penalty=1,
                temperature=0.5
            )
            llm_message = llm_message.replace("<s>", "") # handle <s>

            with open('/scratch2/kat049/Git/Ask-Anything/video_chat2/example/llm_messages.txt', 'a') as file:
                file.write(llm_message + '\n')

    print(f"Answer: {llm_message}")



if __name__== '__main__':
    chat = load_models()
    gradio_stuff(chat)
    