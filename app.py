import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from roboflow import Roboflow
from PIL import Image
import base64

app = dash.Dash(__name__)
server = app.server

# Initialize Roboflow
rf = Roboflow(api_key="lyM2cd6zGDHzF4feo6T0")
project = rf.workspace().project("tea-7zwis")
model = project.version(1).model

rf1 = Roboflow(api_key="lyM2cd6zGDHzF4feo6T0")
project1 = rf1.workspace().project("healthy-and-nonhealthy-leaves")
model1 = project1.version(1).model

image_model = Image.open("ModelFlow.png")
#image_workflow = Image.open("WorkFlow.png")
# Define the three different image processing functions
def process_image1(image_path):
    # Only Tea disease 
        # Infer on the uploaded image
    prediction = model.predict(image_path).json()
    num_leaves = len(prediction['predictions'])
    leaf_classes = [item['class'] for item in prediction['predictions'][0]['predictions']]


    # Save the prediction image
    # model.predict(image_path).save("prediction_dash.jpg")

    # Open the prediction image
    image = Image.open(image_path)

    return num_leaves, image, leaf_classes[0]

def process_image2(image_path):
    # Infer on the uploaded image
    prediction = model1.predict(image_path).json()
    num_leaves = len(prediction['predictions'])
    classes = [item['class'] for item in prediction['predictions']]
    # Save the prediction image
    # model1.predict(image_path).save("prediction_dash.jpg")

    # Open the prediction image
    image = Image.open(image_path)

    return num_leaves, image, classes[0]
    

def process_image3(image_path):
    prediction = model1.predict(image_path).json()
    num_leaves = len(prediction['predictions'])
    classes = [item['class'] for item in prediction['predictions']]
    class_final = classes[0]
    if(classes[0] == 'Non_Healthy_Leaf'):
        prediction1 = model.predict(image_path).json()
        num_leaves = len(prediction1['predictions'])
        leaf_classes = [item['class'] for item in prediction1['predictions'][0]['predictions']]
        class_final = class_final + " and " + leaf_classes[0]
        
    else:
        class_final = classes[0]
    image = Image.open(image_path)
    return num_leaves , image, class_final

# Map the function names to the actual functions
image_processing_functions = {
    "Only Tea leaves disease prediction": process_image1,
    "Only Healthy vs Non-healthy prediction": process_image2,
    "Complete pipeline as described in model diagram above": process_image3,
}

with open("Introduction.txt", "r") as file:
    text_content1 = file.read()

with open("Remaining.txt", "r") as file:
    text_content2 = file.read()

with open("Dialog.txt", "r") as file:
    text_content3 = file.read()
app.layout = html.Div(
    children=[
        html.H1("Pyresearch", style={"font-family": "Arial, sans-serif", "color": " #008000", "margin-bottom": "10px"}),
        html.H2("Tea Leaves Disease Detection", style={"font-family": "Arial, sans-serif", "color": "#008000"}),
        html.H3("", style={"font-family": "Arial, sans-serif", "color": "#999"}),
        html.P(text_content1),
        html.Img(src=image_model, style={'width': '1000px'}),
        html.P(text_content2),
      #  html.Img(src=image_workflow, style={'width': '1000px'}),
        html.P(text_content3),
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select Image")]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px auto",
                "font-family": "Arial, sans-serif",
                "color": "#333",
                "background-color": "#f0f0f0",
            },
            multiple=False,
        ),
        dcc.Dropdown(
            id="function-selector",
            options=[{'label': function_name, 'value': function_name} for function_name in image_processing_functions.keys()],
            value=list(image_processing_functions.keys())[0],  # Default to the first function
            style={"width": "50%", "margin": "10px auto"},
        ),
        html.Div(id="output-image", style={"font-family": "Arial, sans-serif", "color": "#333"}),
    ],
    style={"margin": "20px", "text-align": "center"},
)

@app.callback(
    Output("output-image", "children"),
    Input("upload-image", "contents"),
    Input("function-selector", "value"),
    prevent_initial_call=True,
)
def update_output(contents, selected_function):
    if contents is not None:
        content_type, content_string = contents.split(",")
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(content_string))

        # Call the selected image processing function
        num_leaves, image, leaf_classes = image_processing_functions[selected_function](image_path)

        # Display the prediction results
        output = [
            html.H3(f"Number of leaves detected: {num_leaves}", style={"font-family": "Arial, sans-serif", "color": "#333"}),
            html.H4("Leaf Classes:"),
            html.Ul(leaf_classes),
            html.Img(src=image, style={"width": "50%", "height": "600px"}),
        ]
        return output

    return ""

if __name__ == "__main__":
    app.run_server(debug=True, port=8059)
