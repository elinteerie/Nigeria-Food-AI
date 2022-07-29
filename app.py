import requests
import gradio as gr

# Download human-readable labels for ImageNet.
response = requests.get("https://raw.githubusercontent.com/elinteerie/Nigeria-Food-AI/main/labels.txt")
labels = response.text.split("\n")

def classify_image(inp):
  inp = inp.reshape((-1, 224, 224, 3))
  prediction = model_depl.predict(inp).flatten()
  confidences = {labels[i]: float(prediction[i]) for i in range(14)}
  return confidences



gr.Interface(fn=classify_image, 
             inputs=gr.Image(shape=(224, 224)),
             outputs=gr.Label(num_top_classes=3),
             examples=["/content/egusi sample.jpg", "/content/Ogbono soup.jpg"]).launch()
