import SimpleITK
import numpy as np

from pandas import DataFrame
from scipy.ndimage import center_of_mass, label
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from skimage import transform
import json
from typing import Dict
import training_utils.utils as utils
from training_utils.dataset import CXRNoduleDataset, get_transform
import os
from training_utils.train import train_one_epoch
import itertools
from pathlib import Path
from postprocessing import get_NonMaxSup_boxes, get_weighted_boxes_fusion_boxes_4

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True

class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=False, retrain=False, retest=False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(input_dir),
            output_file = Path(os.path.join(output_dir,'nodules.json'))
        )
        
        #------------------------------- LOAD the model here ---------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_path, self.output_path = input_dir, output_dir
        print('using the device ', self.device)

        num_classes = 2  # 1 class (nodule) + background
        # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        anchor_sizes = ((16,), (32,), (64,), (128,), (192,))
        aspect_ratios = ((0.8, 1.0, 1.25),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if train or retrain:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                              pretrained_backbone=False,
                                                                              min_size=(800, 896, 960, 1024, 1088, 1152, 1216, 1333),
                                                                              max_size=1333)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model.rpn.anchor_generator = rpn_anchor_generator

            if retrain:
                print('loading the model.pth file :')
                self.model.load_state_dict(
                    torch.load(
                        Path("/opt/algorithm/model.pth") if execute_in_docker else Path("model.pth"),
                        map_location=self.device,
                    )
                )
            self.model.to(self.device)

        if retest:
            self.model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                               pretrained_backbone=False,
                                                                               min_size=800, max_size=800)
            self.model2 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                               pretrained_backbone=False,
                                                                               min_size=1333, max_size=1333)

            in_features = self.model1.roi_heads.box_predictor.cls_score.in_features

            self.model1.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model1.rpn.anchor_generator = rpn_anchor_generator

            self.model2.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            self.model2.rpn.anchor_generator = rpn_anchor_generator

            print('loading the retrained model_retrained.pth file')
            self.model1.load_state_dict(
                torch.load(
                    Path("/opt/algorithm/model_retrained_15.pth") if execute_in_docker else Path("model_retrained_15.pth"),
                    map_location=self.device,
                )
            )

            self.model2.load_state_dict(
                torch.load(
                    Path("/opt/algorithm/model_retrained_15.pth") if execute_in_docker else Path("model_retrained_15.pth"),
                    map_location=self.device,
                )
            )
            self.model1.to(self.device)
            self.model2.to(self.device)

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)
            
    # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results. 
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225) 
        is overwritten here, so that it directly returns the predictions without changing the format.
        
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)
        
        # Write resulting candidates to nodules.json for this case
        return scored_candidates
    
   
    
    #--------------------Write your retrain function here ------------
    def train(self, num_epochs = 16):
        '''
        input_dir: Input directory containing all the images to train with
        output_dir: output_dir to write model to.
        num_epochs: Number of epochs for training the algorithm.
        '''
        # Implementation of the pytorch model and training functions is based on pytorch tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        # create training dataset and defined transformations
        self.model.train() 
        input_dir = self.input_path
        dataset = CXRNoduleDataset(input_dir, os.path.join(input_dir, 'metadata.csv'), get_transform(train=True))
        print('training starts ')
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
    
        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=4,
                                                       gamma=0.1)        
        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            print('epoch ', str(epoch),' is running')
            # evaluate on the test dataset
            
            #IMPORTANT: save retrained version frequently.
            print('saving the model')
            torch.save(self.model.state_dict(), os.path.join(self.output_path, 'model_retrained_'+str(epoch)+'.pth'))
      

    def format_to_GC(self, np_prediction, spacing) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation
        
        
        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.
        
        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters. 
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}   
            box['corners']=[]
            x_min, y_min, x_max, y_max = bb*x_y_spacing
            x_min, y_min, x_max, y_max  = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min,  np_prediction['slice'][i]] 
            bottom_right = [x_max, y_min,  np_prediction['slice'][i]]
            top_left = [x_min, y_max,  np_prediction['slice'][i]]
            top_right = [x_max, y_max,  np_prediction['slice'][i]]
            box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            boxes.append(box)
        
        return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={ "major": 1, "minor": 0 })
        
    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d
        
    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        self.model1.eval()
        self.model2.eval()

        image_data = SimpleITK.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        image_data = np.array(image_data)
        
        if len(image_data.shape)==2:
            image_data = np.expand_dims(image_data, 0)
            
        results = []
        # operate on 3D image (CXRs are stacked together)
        for j in range(len(image_data)):
            # Pre-process the image
            image = image_data[j,:,:]
            # The range should be from 0 to 1.
            image = image.astype(np.float32) / np.max(image)  # normalize
            image = np.expand_dims(image, axis=0)
            tensor_image1 = torch.from_numpy(image).to(self.device)#.reshape(1, 1024, 1024)

            width = tensor_image1.shape[-1:]
            tensor_image2 = tensor_image1.flip(-1)
            with torch.no_grad():
                # R_800
                prediction11 = self.model1([tensor_image1.to(self.device)])
                prediction21 = self.model1([tensor_image2.to(self.device)])
                prediction21[0]['boxes'][:, [0, 2]] = torch.tensor(width).to(self.device) - prediction21[0]['boxes'][:, [2, 0]]

                prediction11 = get_NonMaxSup_boxes(prediction11[0])
                prediction21 = get_NonMaxSup_boxes(prediction21[0])

                # R_1333
                prediction12 = self.model2([tensor_image1.to(self.device)])
                prediction22 = self.model2([tensor_image2.to(self.device)])
                prediction22[0]['boxes'][:, [0, 2]] = torch.tensor(width).to(self.device) - prediction22[0]['boxes'][:, [2, 0]]

                prediction12 = get_NonMaxSup_boxes(prediction12[0])
                prediction22 = get_NonMaxSup_boxes(prediction22[0])

            prediction = [get_weighted_boxes_fusion_boxes_4(prediction11, prediction21, prediction12, prediction22, self.device)]

            # convert predictions from tensor to numpy array.
            np_prediction = {str(key):[i.cpu().numpy() for i in val]
                   for key, val in prediction[0].items()}
            np_prediction['slice'] = len(np_prediction['boxes'])*[j]
            results.append(np_prediction)
        
        predictions = self.merge_dict(results)
        data = self.format_to_GC(predictions, spacing)
        print(data)
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    parser.add_argument('--train', action='store_true', help = "Algorithm on train mode.")
    parser.add_argument('--retrain', action='store_true', help = "Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest', action='store_true', help = "Algorithm on evaluate mode after retraining.")

    parsed_args = parser.parse_args()  
    if (parsed_args.train or parsed_args.retrain):# train mode: retrain or train
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, parsed_args.train, parsed_args.retrain, parsed_args.retest).train()
    else:# test mode (test or retest)
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, retest=parsed_args.retest).process()
            
    
   
    
    
    
    
    
