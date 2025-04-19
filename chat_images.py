
import base64
from io import BytesIO
from PIL import Image
import openai

def encode_image(image_path, max_image=512):
    
    '''
        This function encodes an image to a base64 string, resizing it if necessary to fit within a maximum dimension.
    '''
    
    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > max_image:
            scale_factor = max_image / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    

def chat_response(image_path,user_prompt):
    
    '''
        This function generates a chat response by sending an image and user prompt to an AI model.
    '''
    system_prompt = "You are an expert at analyzing images."

    for i,image in enumerate(image_path):
        encoded_image = encode_image(image_path)

    try:
        apiresponse = openai.ChatCompletion.create(
            engine="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                    ],
                },
            ],
            max_tokens=500,
        )

        # Print the response
        response_content = apiresponse.choices[0].message.content
        print("Response:", response_content)

    except Exception as e:
        print("An error occurred:", e)

    return response_content
