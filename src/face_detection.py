'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import logging

from input_feeder import InputFeeder
import numpy as np
import os
from openvino.inference_engine import IECore

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        '''
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0 and self.device=='CPU':
                print("unsupported layers found:{}".format(unsupported_layers))
                if not self.extensions==None:
                    print("Adding cpu_extension")
                    self.plugin.add_extension(self.extensions, self.device)
                    supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                    unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                    if len(unsupported_layers)!=0:
                        print("After adding the extension still unsupported layers found")
                        exit(1)
                    print("After adding the extension the issue is resolved")
                else:
                    print("Give the path of cpu extension")
                    exit(1) 
        '''
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        preprocessed_image = self.preprocess_input(image.copy())
        output = self.exec_net.infer({self.input_name:preprocessed_image})
        coords = self.preprocess_output(output)
        if (len(coords)==0):
            return 0, 0
        coords = coords[0]

        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_image = np.transpose(np.expand_dims(resized_image,axis=0), (0,3,1,2))
        return preprocessed_image

    def preprocess_output(self, outputs, prob_threshold=0.6):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords =[]
        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            if conf>prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return coords


'''
def main():
    
    modelPathDict = {'FaceDetection':"/Users/jeremycohen/Downloads/starter/models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"}
    logger = logging.getLogger()

    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = FaceDetection(modelPathDict['FaceDetection'], "CPU", None)

    #mc = MouseController('medium','fast')
    inputFeeder = InputFeeder("cam")
    inputFeeder.load_data()
    fdm.load_model()
    
    frame_count = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy())
        cv2.imshow("visualization",cv2.resize(croppedFace,(500,500)))

        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        if key==27:
                break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
if __name__ == '__main__':
    main() 
'''