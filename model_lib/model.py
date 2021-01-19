import json
import os

import sys
import ast
import torch
from PIL import Image
from torchvision import models, transforms

from flask_psc_model import ModelBase, load_metadata

# define data directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ROOT_DIR, 'imagenet_classes.txt')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights/resnet101_weights.pth')

class BasicImageClassifier(ModelBase):
    """Example model that predicts the object class of an image."""

    #: load the `model.yaml` metadata file from up the filesystem hierarchy;
    #: this will be used to avoid hardcoding the below filenames in this file
    metadata = load_metadata(__file__)

    #: a list of input filenames; specifying the `input_filenames` attribute is required to configure the model app
    input_filenames = list(metadata.inputs)

    #: a list of output filenames; specifying the `output_filenames` attribute is required to configure the model app
    output_filenames = list(metadata.outputs)

    def __init__(self):
        """Load the model files and do any initialization.

        A single instance of this model class will be reused multiple times to perform inference
        on multiple input files so any slow initialization steps such as reading in a data
        files or loading an inference graph to GPU should be done here.

        This function should require no arguments, or provide appropriate defaults for all arguments.

        NOTE: The `__init__` function and `run` function may not be called from the same thread so extra
        care may be needed if using frameworks such as Tensorflow that make use of thread locals.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = models.resnet101()        
        self.model.load_state_dict(torch.load(WEIGHTS_DIR))
        
        self.model.to(self.device)
        
        self.model.eval()
                
        # labels
        with open(LABELS, 'r') as f:
            self.labels = ast.literal_eval(f.read())
            
        # define data transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        


    def preprocess(self, image):
        
        # do data transformation
        img_t = self.transform(image)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)
        
        return batch_t
    
    def postprocess(self, predictions):
        
        percentage = torch.nn.functional.softmax(predictions, dim=1)[0]

        _, indices = torch.sort(predictions, descending=True)
        top5_preds = [(self.labels[idx.item()], percentage[idx].item()) for idx in indices[0][:5]]
        
        return top5_preds
    
    
    def run(self, input_path, output_path):
        """Run the model on the given input file paths and write to the given output file paths.

        The input files paths followed by the output file paths will be passed into this function as
        positional arguments in the same order as specified in `input_filenames` and `output_filenames`.
        """
        # read in data
        image = Image.open(input_path)
        
        # data preprocessing
        img = self.preprocess(image)
        
        # perform inference
        output = self.model(img)
        
        # post process
        results = self.postprocess(output)
        
        # save output
        results = {'results': results}

        with open(output_path, 'w') as out:
            json.dump(results, out)
        

if __name__ == '__main__':
    # run the model independently from the full application; can be useful for testing
    #
    # to run from the repository root:
    #     python -m model_lib.model /path/to/input.txt /path/to/output.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input data filepath')
    parser.add_argument('output', help='the output results filepath')
    args = parser.parse_args()

    model = BasicImageClassifier()
    model.run(args.input, args.output)
