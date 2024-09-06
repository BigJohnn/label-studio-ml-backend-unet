from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.converter import brush
from unet import UNet
from PIL import Image
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from utils.data_loading import BasicDataset
from uuid import uuid4
import requests
import label_studio_sdk
from utils.dice_score import dice_loss
import os
logger = logging.getLogger(__name__)
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from pathlib import Path

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

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

dir_checkpoint = Path('./checkpoints/')

def train_net(images,masks,net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    
    dataset = BasicDataset(images, masks, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
        net = UNet(n_channels=3, n_classes=2, bilinear=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        MODELPATH='/checkpoints/checkpoint_epoch5.pth'
        logging.info(f'Loading model {MODELPATH}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(MODELPATH, map_location=device))

        self.LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
        self.LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')

        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

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
        # mm=np.where(mm > 127, 255, 0)
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

    def rle_decode(self, rle_data, shape):
        """
        将RLE数据解码为NumPy数组

        Args:
            rle_data: RLE数据，列表或数组形式
            shape: 原始图像的形状

        Returns:
            mask: 解码后的掩码，NumPy数组
        """

        mask = np.zeros(shape, dtype=np.uint8)
        start = 0
        for i in range(0, len(rle_data), 2):
            start = rle_data[i]
            length = rle_data[i+1]
            mask[start : start + length] = 1
        return mask

    
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

        # api_url = 'http://localhost:8080/api/projects/1/tasks'
        # token = 'afcd5aaccf53e3d45e59623c9ecb86b37cb3117e'

        # headers = {'Authorization': f'Token {token}'}
        # response = requests.get(api_url, headers=headers)
        # print('---------',response)

        # If the response is JSON, parse it into a Python object
        # json_data = response.json()
        # print("JSON data:", json_data)

        # images, masks = self.load_data(data)

        # Train the U-Net model (replace with your training loop)
        if event == 'START_TRAINING':
            logger.info("Fitting model")

            # download annotated tasks from Label Studio
            ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
            project = ls.get_project(id=self.project_id)
            tasks = project.get_labeled_tasks()

            logger.info(f"Downloaded {len(tasks)} labeled tasks from Label Studio")

            images=[]
            masks=[]
            for task in tasks:
                for annotation in task['annotations']:
                    if not annotation.get('result') or annotation.get('skipped') or annotation.get('was_cancelled'):
                        continue

                    path = self.get_local_path(task['data']['image'], task_id=task['id'])
                    image = Image.open(path)
                    images.append(image)

                    res = task['annotations'][0]['result'][0]
                    # image.save('test.jpg')
                    width = res['original_width']
                    height = res['original_height']
                    rle = res['value']['rle']

                    # 示例RLE数据
                    shape = (width, height)  # 图像形状

                    # 解码RLE
                    mask = brush.decode_rle(rle)
                    mask = np.reshape(mask, [150, 150, 4])[:, :, 3]
                    masks.append(Image.fromarray(mask/255))

                    # # 创建PIL图像
                    # pil_image = Image.fromarray(mask, mode='L')
                    # # 保存为JPEG
                    # pil_image.save('masks/mask'+str(task['id'])+'.jpg')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = UNet(n_channels=3, n_classes=2, bilinear=False)
            net.to(device=device)
            train_net(images,masks,net,device)

        # Update model version and data information after training
        # self.set('model_version', '0.0.3')  # Update with actual new version
        # self.set('my_data', 'latest')

        print(f'Training completed (if applicable) for event: {event}')

        # # store new data to the cache
        # self.set('my_data', 'd8-31')
        # self.set('model_version', '0.0.2')
        # print(f'New data: {self.get("my_data")}')
        # print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

