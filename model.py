from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from unet import UNet
from PIL import Image
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loading import BasicDataset
from uuid import uuid4

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))
class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
        net = UNet(n_channels=3, n_classes=2, bilinear=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        MODELPATH='/home/hph/Codes/inference_test/checkpoint_epoch5.pth'
        logging.info(f'Loading model {MODELPATH}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(MODELPATH, map_location=device))

        logging.info('Model loaded!')
        self.net=net
        self.device=device

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        path = self.get_local_path(tasks[0]['data']['image'], task_id=tasks[0]['id'])

        image = Image.open(path)
        # image = np.array(image.convert("RGB"))
        mask = predict_img(net=self.net,
                           full_img=image,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=self.device)
        print('====',mask, mask.shape)
        maskIm=mask_to_image(mask)

        print('====',maskIm)
        maskIm.save("mask.jpg")

        results = []
        total_prob = 0
        # for mask, prob in zip(masks, probs):
        # creates a random ID for your label everytime so no chance for errors
        label_id = str(uuid4())[:4]
        # converting the mask from the model to RLE format which is usable in Label Studio
        mm=(np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        print(mm)


        # mm = (mm > 128).astype(np.uint8) * 255
        rle = brush.mask2rle(mm)

        selected_label='Seam'

        from_name, to_name, value = self.get_first_tag_occurence('BrushLabels', 'Image')
        width,height = mask.shape[2],mask.shape[1]
        results.append({
            'id': label_id,
            'from_name': from_name,
            'to_name': to_name,
            'original_width': width,
            'original_height': height,
            'image_rotation': 0,
            'value': {
                'format': 'rle',
                'rle': rle,
                'brushlabels': [selected_label],
            },
            'type': 'brushlabels',
            'readonly': False
        })

        return [{
            'result': results,
            'model_version': self.get('model_version')
            # 'score': total_prob / max(len(results), 1)
        }]

    def load_data(self, data):
        """
        This function loads training data (images and masks) from Label Studio.

        Args:
            data (dict): Payload received from the event.

        Returns:
            tuple: (images, masks)
        """
        # Implement logic to extract image and mask data from Label Studio payload (data)
        # You might need to use data['annotations'] or similar fields
        # Convert data to tensors and normalize as needed
        # images = ...
        # masks = ...
        print(f'data: {data}')
        
        return images, masks

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        images, masks = self.load_data(data)

        # Train the U-Net model (replace with your training loop)
        if event == 'START_TRAINING' or (old_data != 'd8-31' or old_model_version != '0.0.2'):
            model = UNet(in_channels=3, out_channels=1)  # Adjust channels based on your data
            # Define optimizer, loss function, hyperparameters (e.g., learning rate)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.BCEWithLogitsLoss()
            num_epochs = 10  # Adjust training epochs as needed

            for epoch in range(num_epochs):
                # Train loop logic (forward pass, loss calculation, backpropagation)
                for image, mask in zip(images, masks):
                    # ...

                # Update model version and data information after training
                self.set('model_version', '0.0.3')  # Update with actual new version
                self.set('my_data', 'latest')

        print(f'Training completed (if applicable) for event: {event}')

        # # store new data to the cache
        # self.set('my_data', 'd8-31')
        # self.set('model_version', '0.0.2')
        # print(f'New data: {self.get("my_data")}')
        # print(f'New model version: {self.get("model_version")}')

        print(f'event: {event}')


        print('fit() completed successfully.')

